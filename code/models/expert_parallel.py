import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
from .models import BaseModel
from config import Config

class ExpertParallelModel(BaseModel):
    def __init__(self, data_loader, test_loader, dist_config, archive_dir=None, use_saved="no"):
        super().__init__(
            data_loader=data_loader,
            test_loader=test_loader,
            dist_config=dist_config,
            archive_dir=archive_dir,
            use_saved=use_saved,
            parallelism_type="expert_parallel"
        )

    def build_model(self):
        self.num_experts = self.num_compute_nodes

        if self.rank == 0:
            # Rank 0: Gate
            model = nn.Linear(self.input_dim, self.num_experts).to(self.device)
            if self.use_saved == "yes" and self.archive_dir:
                model_path = os.path.join(self.archive_dir, "model", "expert_parallel_gate_model.pt")
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, weights_only=True))
                    print(f"Rank 0: Loaded saved gate model.")
        else:
            # Rank 1-4: Experts
            model = nn.Sequential(
                nn.Linear(self.input_dim, Config.HIDDEN_DIM),
                nn.ReLU(),
                nn.Linear(Config.HIDDEN_DIM, self.num_classes)
            ).to(self.device)
            if self.use_saved == "yes" and self.archive_dir:
                model_path = os.path.join(self.archive_dir, "model", f"expert_parallel_rank{self.rank}_model.pt")
                if os.path.exists(model_path):
                    model.load_state_dict(torch.load(model_path, weights_only=True))
        return model

    def build_optimizer(self):
        if self.rank == 0:
            return torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)
        return torch.optim.SGD(self.model.parameters(), lr=Config.LEARNING_RATE)

    def build_criterion(self):
        if self.rank == 0:
            return nn.CrossEntropyLoss()
        return None

    def train(self, epochs=Config.EPOCHS, time_limit=None):
        start_time = time.time()
        batch_size = self.data_loader.batch_size if self.data_loader else Config.BATCH_SIZE

        # Rank 0: Gate
        if self.rank == 0:
            if self.use_saved != "yes":
                dist.broadcast(torch.tensor([1]), src=0)
                last_log_time = 0

                current_image = 0
                total_batches = len(self.data_loader)
                total_images = total_batches * batch_size * epochs

                epoch = 0
                keep_training = True

                while keep_training:
                    for i, (batch_x, batch_y) in enumerate(self.data_loader):

                        actual_batch_size = batch_x.size(0)
                        current_image += actual_batch_size

                        batch_x, batch_y = batch_x.view(-1, self.input_dim), batch_y.long()

                        dist.broadcast(torch.tensor([1]), src=0)

                        with torch.no_grad():
                            gate_scores = self.model(batch_x)
                            expert_assignments = torch.argmax(gate_scores, dim=1) + 1

                        expert_losses = []
                        for r in range(1, self.world_size):
                            mask = expert_assignments == r
                            sub_x = batch_x[mask]
                            sub_y = batch_y[mask]

                            count = torch.tensor([len(sub_x)])
                            dist.send(count, dst=r)

                            if len(sub_x) > 0:
                                dist.send(sub_x, dst=r)
                                pred_buffer = torch.zeros(len(sub_x), self.num_classes)
                                dist.recv(pred_buffer, src=r)

                                pred_buffer.requires_grad = True
                                loss = nn.CrossEntropyLoss()(pred_buffer, sub_y)
                                loss.backward()
                                expert_losses.append(loss.item())
                                dist.send(pred_buffer.grad, dst=r)

                        self.optimizer.zero_grad()
                        gate_loss = torch.tensor(expert_losses).sum()
                        gate_loss.requires_grad = True
                        gate_loss.backward()
                        self.optimizer.step()

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
                if self.archive_dir and os.path.isdir(self.archive_dir):
                    model_dir = os.path.join(self.archive_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(model_dir, "expert_parallel_gate_model.pt"))

                self.train_logger.log_training_result(
                    training_time=training_time,
                    total_images=current_image,
                    epochs=epoch,
                    batch_size=batch_size
                )

        # Rank 1-4: Experts
        else:
            mode_signal = torch.tensor([0])
            dist.broadcast(mode_signal, src=0)

            if mode_signal.item() == 1:
                step_signal = torch.tensor([0])
                last_log_time = 0
                while True:
                    dist.broadcast(step_signal, src=0)
                    if step_signal.item() == 0:
                        break

                    count_tensor = torch.tensor([0])
                    dist.recv(count_tensor, src=0)
                    curr_count = count_tensor.item()

                    if curr_count > 0:
                        input_data = torch.zeros(curr_count, self.input_dim)
                        dist.recv(input_data, src=0)

                        input_data.requires_grad = True
                        self.optimizer.zero_grad()
                        output = self.model(input_data)

                        dist.send(output, dst=0)

                        grad_in = torch.zeros(curr_count, self.num_classes)
                        dist.recv(grad_in, src=0)

                        output.backward(grad_in)
                        self.optimizer.step()

                        current_time = time.time()
                        if current_time - last_log_time >= 1.0:
                            self.collect_system_logs()
                            last_log_time = current_time

                self.collect_system_logs()

                if self.archive_dir and os.path.isdir(self.archive_dir):
                    model_dir = os.path.join(self.archive_dir, "model")
                    os.makedirs(model_dir, exist_ok=True)
                    torch.save(self.model.state_dict(), os.path.join(model_dir, f"expert_parallel_rank{self.rank}_model.pt"))

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
                    gate_scores = self.model(batch_xt)
                    expert_assignments = torch.argmax(gate_scores, dim=1) + 1
                    batch_preds = torch.zeros(actual_batch_size, self.num_classes)

                    for r in range(1, self.world_size):
                        mask = expert_assignments == r
                        sub_x = batch_xt[mask]

                        count = torch.tensor([len(sub_x)])
                        dist.send(count, dst=r)

                        if len(sub_x) > 0:
                            dist.send(sub_x, dst=r)
                            pred_buffer = torch.zeros(len(sub_x), self.num_classes)
                            dist.recv(pred_buffer, src=r)
                            batch_preds[mask] = pred_buffer

                    acc = (batch_preds.argmax(dim=1) == batch_yt).sum().item()
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

                        count_tensor = torch.tensor([0])
                        dist.recv(count_tensor, src=0)
                        curr_count = count_tensor.item()

                        if curr_count > 0:
                            input_data = torch.zeros(curr_count, self.input_dim)
                            dist.recv(input_data, src=0)
                            output = self.model(input_data)
                            dist.send(output, dst=0)
            dist.barrier()