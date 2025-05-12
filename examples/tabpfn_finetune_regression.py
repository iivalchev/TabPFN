"""An example for finetuning TabPFN on the Covertype dataset."""
# import json
from functools import partial
# from pathlib import Path

import sklearn
import torch
from sklearn.model_selection import train_test_split
from torch.optim import Adam
from torch.utils.data import DataLoader
from tqdm import tqdm

from tabpfn import TabPFNRegressor
from tabpfn.utils import collate_for_tabpfn_dataset


def eval_test(reg, my_dl_test, lossfn):
    with torch.no_grad():
        loss_sum = 0.0
        num_batches = 0
        for data_batch in my_dl_test:
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            reg.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = reg.predict_from_preprocessed(X_tests)
            loss_sum += lossfn(preds, y_tests).item()
            num_batches += 1
        return loss_sum / num_batches

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    do_epochs = 100

    # Load Diabetes Dataset (regression)
    data_frame_x, data_frame_y = sklearn.datasets.load_diabetes(return_X_y=True)
    splitfn = partial(train_test_split, test_size=0.3, random_state=42)
    X_train, X_test, y_train, y_test = splitfn(data_frame_x, data_frame_y)

    reg = TabPFNRegressor(n_estimators=4)

    datasets_list = reg.get_preprocessed_datasets(X_train, y_train, splitfn)
    datasets_list_test = reg.get_preprocessed_datasets(X_test, y_test, splitfn)
    my_dl_train = DataLoader(datasets_list, batch_size=1, collate_fn=collate_for_tabpfn_dataset)
    my_dl_test = DataLoader(
        datasets_list_test, batch_size=1, collate_fn=collate_for_tabpfn_dataset
    )

    def lossfn(logits_per_config: list[torch.Tensor], y_true: torch.Tensor):
        # logits_per_config is a list (per EnsembleConfig) of [batch, seq, num_bars]
        return torch.stack(
            [
                reg.renormalized_criterion_(logits.transpose(0, 1), y_true)
                for logits in logits_per_config
            ]
        ).mean()

    optim_impl = Adam(reg.model_.parameters(), lr=1e-5)
    loss_batches = []

    loss_test = eval_test(reg, my_dl_test, lossfn)
    print("Initial loss:", loss_test)
    for epoch in range(do_epochs):
        for data_batch in tqdm(my_dl_train):
            optim_impl.zero_grad()
            X_trains, X_tests, y_trains, y_tests, cat_ixs, confs = data_batch
            reg.fit_from_preprocessed(X_trains, y_trains, cat_ixs, confs)
            preds = reg.predict_from_preprocessed(X_tests)
            loss = lossfn(preds, y_tests.to(device))
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                reg.model_.parameters(),
                max_norm=1.0,
                error_if_nonfinite=False,
            ).item()
            print(
                f"grad norm: {grad_norm}; grad norm (after clipping): {torch.tensor([p.grad.norm() for p in reg.model_.parameters()]).norm()}"
            )
            optim_impl.step()
            print("Train Loss:", loss)

        loss_test = eval_test(reg, my_dl_test, lossfn)
        print(f"---- EPOCH {epoch}: ----")
        print("Test Loss:", loss_test)

        # loss_batches.append(loss_test)
        # acc_batches.append(res_acc)
        # with Path("finetune.json").open(mode="w") as file:
        #     json.dump({"loss": loss_batches, "acc": acc_batches}, file)

