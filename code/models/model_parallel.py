import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from .models import BaseModel
from config import Config

class ModelParallelModel(BaseModel):
    def __init__(self, data_loader, test_loader, dist_config, archive_dir=None, use_saved="no"):
        super().__init__(
            data_loader=data_loader,
            test_loader=test_loader,
            dist_config=dist_config,
            archive_dir=archive_dir,
            use_saved=use_saved,
            parallelism_type="model_parallel"
        )

        self.compute_group = dist.new_group(list(range(1, self.world_size)))

    def build_model(self):
        self.classes_per_node = (self.num_classes + self.num_compute_nodes - 1) // self.num_compute_nodes
        hidden_size_per_node = Config.HIDDEN_DIM // self.world_size

        model = nn.Sequential(
            nn.Linear(self.input_dim, hidden_size_per_node),
            nn.ReLU(),
            nn.Linear(hidden_size_per_node, self.classes_per_node)
        ).to(self.device)

        return model

    def build_optimizer(self):
        if self.rank == 0:
            return None
        return torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

    def build_criterion(self):
        if self.rank == 0:
            return None
        return nn.CrossEntropyLoss()

    def train(self, epochs=Config.EPOCHS, time_limit=None):
        start_time = time.time()
        batch_size = self.data_loader.batch_size if self.data_loader else Config.BATCH_SIZE

        # Rank 0: Master
        if self.rank == 0:
            if self.use_saved != "yes":
                dist.broadcast(torch.tensor([1]), src=0)
                last_log_time = 0
                current_image = 0
                total_batches = len(self.data_loader)
                total_images = total_batches * batch_size * epochs

                epoch = 0
                keep_training = True

                buffer_x = []
                buffer_y = []

                while keep_training:
                    for i, (batch_x, batch_y) in enumerate(self.data_loader):
                        actual_batch_size = batch_x.size(0)
                        current_image += actual_batch_size

                        batch_x, batch_y = batch_x.view(-1, self.input_dim), batch_y.long()
                        buffer_x.append(batch_x)
                        buffer_y.append(batch_y)

                        x_cat = torch.cat(buffer_x)
                        y_cat = torch.cat(buffer_y)

                        while len(x_cat) >= batch_size:
                            chunk_x = x_cat[:batch_size]
                            chunk_y = y_cat[:batch_size]

                            dist.broadcast(torch.tensor([1]), src=0)
                            dist.broadcast(chunk_x, src=0)
                            dist.broadcast(chunk_y, src=0)

                            current_time = time.time()
                            if current_time - last_log_time >= 1.0:
                                self.train_logger.log_live_progress(current_image, total_images, current_time - start_time, time_limit if time_limit else 0)
                                self.collect_system_logs()
                                last_log_time = current_time

                            x_cat = x_cat[batch_size:]
                            y_cat = y_cat[batch_size:]

                            if time_limit and (current_time - start_time) >= time_limit:
                                keep_training = False
                                break

                        if not keep_training:
                            break

                        buffer_x = [x_cat]
                        buffer_y = [y_cat]

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

        # Rank 1-4: Compute
        else:
            mode_signal = torch.tensor([0])
            dist.broadcast(mode_signal, src=0)

            if mode_signal.item() == 1:
                batch_x = torch.zeros(batch_size, self.input_dim)
                batch_y = torch.zeros(batch_size, dtype=torch.long)
                step_signal = torch.tensor([0])
                last_log_time = 0

                while True:
                    dist.broadcast(step_signal, src=0)
                    if step_signal.item() == 0:
                        break

                    dist.broadcast(batch_x, src=0)
                    dist.broadcast(batch_y, src=0)

                    self.optimizer.zero_grad()
                    local_out = self.model(batch_x)

                    gathered = [torch.zeros(batch_size, self.classes_per_node) for _ in range(self.num_compute_nodes)]
                    dist.all_gather(gathered, local_out.data, group=self.compute_group)

                    full_parts = []
                    for r, tensor in enumerate(gathered):
                        if r == (self.rank - 1):
                            full_parts.append(local_out)
                        else:
                            full_parts.append(tensor)

                    full_logits = torch.cat(full_parts, dim=1)[:, :self.num_classes]
                    loss = self.criterion(full_logits, batch_y)
                    loss.backward()
                    self.optimizer.step()

                    current_time = time.time()
                    if current_time - last_log_time >= 1.0:
                        self.collect_system_logs()
                        last_log_time = current_time

                self.collect_system_logs()

                if self.archive_dir and os.path.isdir(self.archive_dir):
                    model_dir = os.path.join(self.archive_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(model_dir, f"model_parallel_rank{self.rank}_model.pt"))

    def test(self):
        test_start = time.time()
        batch_size = self.test_loader.batch_size if self.test_loader else Config.BATCH_SIZE

        if self.rank == 0:
            dist.broadcast(torch.tensor([2]), src=0)
            correct = 0
            total = 0

            with torch.no_grad():
                buffer_x, buffer_y = [], []
                for batch_xt, batch_yt in self.test_loader:
                    batch_xt, batch_yt = batch_xt.view(-1, self.input_dim), batch_yt.long()

                    buffer_x.append(batch_xt)
                    buffer_y.append(batch_yt)

                    x_cat = torch.cat(buffer_x)
                    y_cat = torch.cat(buffer_y)

                    while len(x_cat) >= batch_size:
                        chunk_x = x_cat[:batch_size]
                        chunk_y = y_cat[:batch_size]

                        dist.broadcast(torch.tensor([1]), src=0)
                        dist.broadcast(chunk_x, src=0)

                        preds = []
                        for r in range(1, self.world_size):
                            buffer = torch.zeros(batch_size, self.classes_per_node)
                            dist.recv(buffer, src=r)
                            preds.append(buffer)

                        full_logits = torch.cat(preds, dim=1)[:, :self.num_classes]
                        acc = (full_logits.argmax(dim=1) == chunk_y).sum().item()
                        correct += acc
                        total += batch_size

                        x_cat = x_cat[batch_size:]
                        y_cat = y_cat[batch_size:]

                    buffer_x = [x_cat]
                    buffer_y = [y_cat]

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
                batch_x = torch.zeros(batch_size, self.input_dim)
                step_signal = torch.tensor([0])

                with torch.no_grad():
                    while True:
                        dist.broadcast(step_signal, src=0)
                        if step_signal.item() == 0:
                            break

                        dist.broadcast(batch_x, src=0)
                        local_out = self.model(batch_x)
                        dist.send(local_out, dst=0)

            dist.barrier()