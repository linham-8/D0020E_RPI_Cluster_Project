import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from .models import BaseModel
from config import Config

class PipelineParallelModel(BaseModel):
    def __init__(self, data_loader, test_loader, dist_config, archive_dir=None, use_saved="no"):
        super().__init__(
            data_loader=data_loader,
            test_loader=test_loader,
            dist_config=dist_config,
            archive_dir=archive_dir,
            use_saved=use_saved,
            parallelism_type="pipeline_parallel"
        )

    def build_model(self):
        self.hidden_dim = Config.HIDDEN_DIM
        if self.rank == 1:
            model = nn.Sequential(
                nn.Linear(self.input_dim, self.hidden_dim),
                nn.ReLU()
            ).to(self.device)

        elif self.rank == self.world_size - 1:
            model = nn.Linear(self.hidden_dim, self.num_classes).to(self.device)

        else:
            model = nn.Sequential(
                nn.Linear(self.hidden_dim, self.hidden_dim),
                nn.ReLU()
            ).to(self.device)

        if self.use_saved == "yes" and self.archive_dir and self.rank > 0:
            model_path = os.path.join(self.archive_dir, "model", f"pipeline_parallel_rank{self.rank}_model.pt")
            if os.path.exists(model_path):
                model.load_state_dict(torch.load(model_path, weights_only=True))

        return model

    def build_optimizer(self):
        if self.rank == 0:
            return None
        return torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

    def build_criterion(self):
        if self.rank == 0:
            return nn.CrossEntropyLoss()
        return None

    def train(self, epochs=Config.EPOCHS, time_limit=None):
        start_time = time.time()
        batch_size = self.data_loader.batch_size if self.data_loader else Config.BATCH_SIZE
        num_microbatches = 4

        # Rank 0: Orchestrator
        if self.rank == 0:
            if self.use_saved != "yes":
                dist.broadcast(torch.tensor([1]), src=0)

                current_image = 0
                total_batches = len(self.data_loader)
                total_images = total_batches * batch_size * epochs

                last_log_time = 0

                epoch = 0
                keep_training = True

                while keep_training:
                    for i, (batch_x, batch_y) in enumerate(self.data_loader):
                        actual_batch_size = batch_x.size(0)
                        current_image += actual_batch_size

                        batch_x, batch_y = batch_x.view(-1, self.input_dim), batch_y.long()

                        micro_x = torch.chunk(batch_x, num_microbatches, dim=0)
                        micro_y = torch.chunk(batch_y, num_microbatches, dim=0)
                        num_mbs = len(micro_x)

                        dist.broadcast(torch.tensor([num_mbs]), src=0)

                        mb_sizes = torch.tensor([mb.size(0) for mb in micro_x], dtype=torch.long)
                        dist.broadcast(mb_sizes, src=0)

                        reqs = []
                        preds_list = []

                        for m in range(num_mbs):
                            reqs.append(dist.isend(micro_x[m], dst=1))

                        for m in range(num_mbs):
                            preds = torch.zeros(mb_sizes[m].item(), self.num_classes)
                            reqs.append(dist.irecv(preds, src=self.world_size - 1))
                            preds_list.append(preds)

                        for req in reqs:
                            req.wait()

                        grad_reqs = []
                        for m in range(num_mbs):
                            preds = preds_list[m]
                            preds.requires_grad = True
                            loss = self.criterion(preds, micro_y[m]) / num_mbs

                            grad_initial = torch.autograd.grad(loss, preds)[0]
                            grad_reqs.append(dist.isend(grad_initial, dst=self.world_size - 1))

                        for req in grad_reqs:
                            req.wait()

                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            self.train_logger.log_live_progress(current_image, total_images, current_time - start_time, time_limit if time_limit else 0)
                            self.collect_system_logs()
                            last_log_time = current_time

                        if time_limit and (current_time - start_time) >= time_limit:
                            keep_training = False
                            break

                    if not keep_training:
                        break

                    epoch += 1
                    if not time_limit and epoch >= epochs:
                        keep_training = False

                dist.broadcast(torch.tensor([0]), src=0)

                self.train_logger.log_live_progress(current_image, total_images, time_limit if time_limit else time.time() - start_time, time_limit if time_limit else 0, is_finished=True)
                self.collect_system_logs()

            else:
                dist.broadcast(torch.tensor([0]), src=0)

            training_time = time.time() - start_time
            print(f"Total Training Time: {training_time:.2f}s")

            if self.use_saved != "yes":
                self.train_logger.log_training_result(
                    training_time=training_time,
                    total_images=total_batches * batch_size * epochs,
                    epochs=epochs,
                    batch_size=batch_size
                )

        # Rank 1-4: Stages
        else:
            mode_signal = torch.tensor([0])
            dist.broadcast(mode_signal, src=0)

            if mode_signal.item() == 1:
                last_log_time = 0
                while True:
                    num_mbs_tensor = torch.tensor([0])
                    dist.broadcast(num_mbs_tensor, src=0)
                    num_mbs = num_mbs_tensor.item()

                    if num_mbs == 0:
                        break

                    mb_sizes = torch.zeros(num_mbs, dtype=torch.long)
                    dist.broadcast(mb_sizes, src=0)

                    self.optimizer.zero_grad()
                    inputs = []
                    outputs = []

                    for m in range(num_mbs):
                        mb_sz = mb_sizes[m].item()

                        if self.rank == 1:
                            input_data = torch.zeros(mb_sz, self.input_dim)
                            dist.recv(input_data, src=0)
                        else:
                            input_data = torch.zeros(mb_sz, self.hidden_dim)
                            dist.recv(input_data, src=self.rank - 1)

                        input_data.requires_grad = True
                        output = self.model(input_data)

                        if self.rank == self.world_size - 1:
                            dist.send(output, dst=0)
                        else:
                            dist.send(output, dst=self.rank + 1)

                        inputs.append(input_data)
                        outputs.append(output)

                    for m in range(num_mbs):
                        mb_sz = mb_sizes[m].item()

                        if self.rank == self.world_size - 1:
                            grad_in = torch.zeros(mb_sz, self.num_classes)
                            dist.recv(grad_in, src=0)
                        else:
                            grad_in = torch.zeros(mb_sz, self.hidden_dim)
                            dist.recv(grad_in, src=self.rank + 1)

                        output = outputs[m]
                        input_data = inputs[m]

                        output.backward(grad_in)

                        if self.rank > 1:
                            dist.send(input_data.grad, dst=self.rank - 1)

                    self.optimizer.step()

                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        self.collect_system_logs()
                        last_log_time = current_time

                self.collect_system_logs()

                if self.archive_dir and os.path.isdir(self.archive_dir):
                    model_dir = os.path.join(self.archive_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(model_dir, f"pipeline_parallel_rank{self.rank}_model.pt"))

    def test(self):
        test_start = time.time()
        batch_size = self.test_loader.batch_size if self.test_loader else Config.BATCH_SIZE

        if self.rank == 0:
            dist.broadcast(torch.tensor([2]), src=0)
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_xt, batch_yt in self.test_loader:
                    actual_batch_size = batch_xt.size(0)
                    batch_xt, batch_yt = batch_xt.view(-1, self.input_dim), batch_yt.long()

                    dist.broadcast(torch.tensor([1]), src=0)
                    dist.broadcast(torch.tensor([actual_batch_size]), src=0)
                    dist.send(batch_xt, dst=1)

                    preds = torch.zeros(actual_batch_size, self.num_classes)
                    dist.recv(preds, src=self.world_size - 1)

                    acc = (preds.argmax(dim=1) == batch_yt).sum().item()
                    correct += acc
                    total += actual_batch_size

            test_time = time.time() - test_start
            dist.broadcast(torch.tensor([0]), src=0)

            final_acc = (correct / total) * 100 if total > 0 else 0
            print(f"Accuracy: {final_acc:.2f}%")

            self.test_logger.log_test_result(
                test_time=test_time,
                accuracy=final_acc,
                total_test_images=total,
                batch_size=batch_size
            )

            dist.barrier()

        else:
            mode_signal = torch.tensor([0])
            dist.broadcast(mode_signal, src=0)

            if mode_signal.item() == 2:
                step_signal = torch.tensor([0])
                with torch.no_grad():
                    while True:
                        dist.broadcast(step_signal, src=0)
                        if step_signal.item() == 0:
                            break

                        bs_tensor = torch.tensor([0])
                        dist.broadcast(bs_tensor, src=0)
                        actual_batch_size = bs_tensor.item()

                        if self.rank == 1:
                            input_data = torch.zeros(actual_batch_size, self.input_dim)
                            dist.recv(input_data, src=0)
                        else:
                            input_data = torch.zeros(actual_batch_size, self.hidden_dim)
                            dist.recv(input_data, src=self.rank - 1)

                        output = self.model(input_data)

                        if self.rank == self.world_size - 1:
                            dist.send(output, dst=0)
                        else:
                            dist.send(output, dst=self.rank + 1)

            dist.barrier()