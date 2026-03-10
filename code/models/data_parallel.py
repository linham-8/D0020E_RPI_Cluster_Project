import os
import time
from datetime import datetime
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from .models import BaseModel
from config import Config

class SimpleLinearModel(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(input_dim, num_classes)

    def forward(self, x):
        x = self.flatten(x)
        return self.linear(x)

class DataParallelModel(BaseModel):
    def __init__(self, data_loader, test_loader, dist_config, archive_dir=None, use_saved="no"):
        super().__init__(
            data_loader=data_loader,
            test_loader=test_loader,
            dist_config=dist_config,
            archive_dir=archive_dir,
            use_saved=use_saved,
            parallelism_type="data_parallel"
        )

    def build_model(self):
        model = SimpleLinearModel(self.input_dim, self.num_classes).to(self.device)

        if self.use_saved == "yes" and self.archive_dir:
            model_path = os.path.join(self.archive_dir, "model", f"{self.parallelism_type}.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=True))
                if self.dist_config['rank'] == 0:
                    print(f"Rank 0: Loaded saved model from {model_path}")

        return DDP(model)

    def build_optimizer(self):
        return torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

    def build_criterion(self):
        return nn.CrossEntropyLoss()

    def train(self, epochs=Config.EPOCHS, time_limit=None):

        if self.use_saved == "yes":

            return

        start_time = time.time()
        last_log_time = 0

        current_image = 0

        total_batches = len(self.data_loader)
        total_images = total_batches * self.data_loader.batch_size * self.world_size * epochs

        epoch = 0
        keep_training = True

        while keep_training:
            if hasattr(self.data_loader.sampler, "set_epoch"):
                self.data_loader.sampler.set_epoch(epoch)

            for i, (batch_x, batch_y) in enumerate(self.data_loader):
                actual_batch_size = batch_x.size(0)

                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                self.optimizer.zero_grad(set_to_none=True)
                outputs = self.model(batch_x)
                loss = self.criterion(outputs, batch_y)
                loss.backward()
                self.optimizer.step()

                if self.rank == 0:
                    current_image += actual_batch_size * self.world_size

                current_time = time.time()
                if current_time - last_log_time >= 1.0:
                    if self.rank == 0:
                        self.train_logger.log_live_progress(current_image, total_images, current_time - start_time, time_limit if time_limit else 0)

                    self.collect_system_logs()
                    last_log_time = current_time

                stop_signal = torch.tensor([0], device=self.device)
                if time_limit and (current_time - start_time) >= time_limit:
                    stop_signal.fill_(1)

                dist.all_reduce(stop_signal, op=dist.ReduceOp.MAX)

                if stop_signal.item() == 1:
                    keep_training = False
                    break

            if not keep_training:
                break

            epoch += 1
            if not time_limit and epoch >= epochs:
                keep_training = False

        if self.rank == 0:
            self.train_logger.log_live_progress(current_image, total_images, time_limit if time_limit else time.time() - start_time, time_limit if time_limit else 0, is_finished=True)

        self.collect_system_logs()

        if self.rank == 0:
            training_time = time.time() - start_time
            print(f"Total Training Time: {training_time:.2f}s")

            if self.archive_dir and os.path.isdir(self.archive_dir):
                model_dir = os.path.join(self.archive_dir, "model")
                os.makedirs(model_dir, exist_ok=True)
                torch.save(self.model.module.state_dict(), os.path.join(model_dir, f"{self.parallelism_type}.pt"))

            self.train_logger.log_training_result(
                training_time=training_time,
                total_images=current_image,
                epochs=epoch,
                batch_size=self.data_loader.batch_size * self.world_size
            )

    def test(self):
        test_start = time.time()
        correct = 0
        total = 0

        with torch.no_grad():
            for batch_xt, batch_yt in self.test_loader:
                batch_xt, batch_yt = batch_xt.to(self.device), batch_yt.to(self.device)
                outputs = self.model(batch_xt)

                acc = (outputs.argmax(dim=1) == batch_yt).sum().item()
                correct += acc
                total += batch_yt.size(0)

        test_time = time.time() - test_start

        if self.rank == 0:
            final_acc = (correct / total * 100) if total > 0 else 0
            print(f"Accuracy: {final_acc:.2f}%")
            self.test_logger.log_test_result(
                test_time=test_time,
                accuracy=final_acc,
                total_test_images=total,
                batch_size=self.test_loader.batch_size
            )