"""Optimizer construction and optional gradient clipping for DuSafe."""

import torch


class GradClipWrapper:
    def __init__(
        self,
        optimizer,
        max_norm=None,
        clip_value=None,
        record_stats=False,
    ):
        self._optimizer = optimizer
        self.max_norm = max_norm
        self.clip_value = clip_value
        self.record_stats = bool(record_stats)
        self.last_pre_clip_grad_norm = float("nan")
        self.last_post_clip_grad_norm = float("nan")
        self.last_pre_clip_grad_abs_max = float("nan")
        self.last_post_clip_grad_abs_max = float("nan")
        self._gradients_prepared = False
        self._prepared_total_norm = None

    @property
    def param_groups(self):
        return self._optimizer.param_groups

    @property
    def state(self):
        return self._optimizer.state

    def state_dict(self):
        return self._optimizer.state_dict()

    def load_state_dict(self, state):
        return self._optimizer.load_state_dict(state)

    def zero_grad(self, *args, **kwargs):
        self._gradients_prepared = False
        self._prepared_total_norm = None
        return self._optimizer.zero_grad(*args, **kwargs)

    def prepare_gradients_for_step(self):
        """Clip once and return the pre-clip norm for a finite check.

        DuSafe checks gradients before committing an optimizer transaction.
        Norm clipping already computes the same global norm, so the common
        norm-only production path can reuse it instead of reducing every
        gradient twice.  Value clipping keeps the historical step path because
        it can turn infinities into finite values before norm clipping.
        """

        if self.clip_value is not None or self.max_norm is None:
            return None
        if self._gradients_prepared:
            return self._prepared_total_norm
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        if not parameters:
            return None
        if self.record_stats:
            pre_norm, pre_max = self._stats(parameters)
            self.last_pre_clip_grad_norm = pre_norm
            self.last_pre_clip_grad_abs_max = pre_max
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        if self.record_stats:
            post_norm, post_max = self._stats(parameters)
            self.last_post_clip_grad_norm = post_norm
            self.last_post_clip_grad_abs_max = post_max
        self._gradients_prepared = True
        self._prepared_total_norm = total_norm.detach()
        return self._prepared_total_norm

    @staticmethod
    def _stats(parameters):
        gradients = [
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None and parameter.grad.numel()
        ]
        if not gradients:
            return 0.0, 0.0
        per_tensor_norms = torch._foreach_norm(gradients, 2.0)
        total_norm = torch.linalg.vector_norm(
            torch.stack([value.float() for value in per_tensor_norms]), 2.0
        )
        absolute_max = torch.stack(
            [gradient.detach().abs().max().float() for gradient in gradients]
        ).max()
        # One device synchronization replaces two synchronizations per
        # parameter in the historical diagnostics path.
        return tuple(float(value) for value in torch.stack((total_norm, absolute_max)).cpu())

    def step(self, closure=None):
        if self._gradients_prepared:
            # ``prepare_gradients_for_step`` already enumerated and clipped the
            # exact gradient set consumed by the wrapped optimizer.  Avoid a
            # second Python traversal of every parameter on the common DuSafe
            # path; the wrapped step and cleanup semantics stay unchanged.
            try:
                return self._optimizer.step(closure)
            finally:
                self._gradients_prepared = False
                self._prepared_total_norm = None
        parameters = [
            parameter
            for group in self.param_groups
            for parameter in group["params"]
            if parameter.grad is not None
        ]
        gradients_prepared = bool(self._gradients_prepared)
        if parameters and self.record_stats and not gradients_prepared:
            (
                self.last_pre_clip_grad_norm,
                self.last_pre_clip_grad_abs_max,
            ) = self._stats(parameters)
        if parameters and not gradients_prepared:
            if self.clip_value is not None:
                torch.nn.utils.clip_grad_value_(parameters, self.clip_value)
            if self.max_norm is not None:
                torch.nn.utils.clip_grad_norm_(parameters, self.max_norm)
        if parameters and self.record_stats and not gradients_prepared:
            (
                self.last_post_clip_grad_norm,
                self.last_post_clip_grad_abs_max,
            ) = self._stats(parameters)
        try:
            return self._optimizer.step(closure)
        finally:
            self._gradients_prepared = False
            self._prepared_total_norm = None


def build_optimizer(hparams):
    def make_optimizer(parameters):
        method = str(hparams["optim_method"]).lower()
        if method == "adam":
            optimizer = torch.optim.Adam(
                parameters,
                lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"],
            )
        elif method == "sgd":
            optimizer = torch.optim.SGD(
                parameters,
                lr=hparams["learning_rate"],
                weight_decay=hparams["weight_decay"],
                momentum=hparams.get("momentum", 0.9),
            )
        else:
            raise NotImplementedError(f"Unknown optimizer: {method}")

        max_norm = float(hparams.get("grad_clip", 0) or 0)
        clip_value = hparams.get("grad_clip_value")
        if clip_value is not None:
            clip_value = float(clip_value)
        if max_norm > 0 or clip_value is not None:
            return GradClipWrapper(
                optimizer,
                max_norm=max_norm if max_norm > 0 else None,
                clip_value=clip_value,
                record_stats=bool(
                    hparams.get("record_optimizer_diagnostics", False)
                ),
            )
        return optimizer

    return make_optimizer


__all__ = ["GradClipWrapper", "build_optimizer"]
