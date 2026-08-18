# LightsclineCompute API Reference

Import:

```python
from lightscline.lightscline import LightsclineCompute
```

`LightsclineCompute` is the main entry point for the pilot release.

---

## Data conventions

### Training data (`__init__`)


| Layout | Shape                                | Description                                                    |
| ------ | ------------------------------------ | -------------------------------------------------------------- |
| 2D     | `(n_classes, n_samples)`             | One channel per class; each row is one time series.            |
| 3D     | `(n_classes, n_channels, n_samples)` | Multiple channels per class; each channel is a list of floats. |


- Values must be **floats** (not bare integers at the leaf level).
- The first dimension is the number of **classes** (or separate chunks of data). If `labels` is omitted, labels default to `0, 1, 2, ...` by class index.
- You may have more rows than distinct classes when discontinuous segments are stored separately; provide an explicit `labels` list in that case.
- Training data is not required when `evaluate_only=True` (see `__init__` and `load_model`).

### Sampling frequency (`fs`)


| Form        | Usage                                               |
| ----------- | --------------------------------------------------- |
| `int`       | Same Hz for every channel.                          |
| `list[int]` | One sampling rate per channel, e.g. `[1700, 1200]`. |


Fractional sampling rates are not supported.

### Inference data (`predict`)


| Layout | Shape                                    |
| ------ | ---------------------------------------- |
| 1D     | `(n_samples,)` — single channel          |
| 2D     | `(n_channels, n_samples)` — multichannel |


Prediction uses the same `window_time`, `per_reduction`, and `fs` as the last successful `reduce_and_train` call (or the values restored by `load_model`). Each window receives one predicted class label.

---

## `LightsclineCompute.__init__`

Create a compute instance for one dataset, or an inference-only instance for `load_model`.

```python
ls = LightsclineCompute(data, fs, labels=None, evaluate_only=False)
```

### Parameters


| Name            | Type                  | Description                                                                                          |
| --------------- | --------------------- | ---------------------------------------------------------------------------------------------------- |
| `data`          | `list`, optional      | 2D or 3D nested lists of floats (see above). Required unless `evaluate_only=True`.                   |
| `fs`            | `int` or `list[int]`  | Sampling frequency (Hz) per channel. Required unless `evaluate_only=True`.                           |
| `labels`        | `list[int]`, optional | Class label for each row in `data`. Defaults to `0..n-1`.                                            |
| `evaluate_only` | `bool`                | If `True`, skip training setup and prepare the instance to `load_model()`. Defaults to `False`.      |


### Raises

- `ValueError` — `evaluate_only` is `False` and `data`/`fs` are missing, data is not 2D/3D floats, or `fs` has an invalid type.

### Inference-only instance

```python
ls = LightsclineCompute(evaluate_only=True)
ls.load_model("model.pkl")
labels = ls.predict(X_test)
```

---

## `LightsclineCompute.reduce_and_train`

Window the dataset, reduces each window, and train the built-in feedforward classifier. This is the primary training entry point for the pilot release.

```python
ls.reduce_and_train(
    per_reduction=90,
    window_time=0.2,
    data_aug_multiplier=2,
    layers=(10, 10),
    learning_rate=0.005,
    n_iters=1000,
    verbose=False,
    weight_decay=0.0001,
    val_size=0.15,
)
```

### Parameters

**Windowing and reduction**


| Name                  | Type                     | Default | Description                                                                                     |
| --------------------- | ------------------------ | ------- | ----------------------------------------------------------------------------------------------- |
| `per_reduction`       | `int`                    | `90`    | Percentage of windows to drop via undersampling (0–100).                                        |
| `window_time`         | `float`                  | `0.2`   | Window length in seconds.                                                                       |
| `data_aug_multiplier` | `float` or `list[float]` | `2`     | Controls window overlap; higher values yield more windows. Use a list to set a value per class. |


**Training**


| Name            | Type         | Default    | Description                                                        |
| --------------- | ------------ | ---------- | ------------------------------------------------------------------ |
| `layers`        | `tuple[int]` | `(10, 10)` | Hidden layer widths; output size is inferred from training labels. |
| `learning_rate` | `float`      | `0.005`    | learning rate.                                                     |
| `n_iters`       | `int`        | `1000`     | Training iterations (epochs).                                      |
| `verbose`       | `bool`       | `False`    | If `True`, prints loss every 50 iterations.                        |
| `weight_decay`  | `float`      | `0.0001`   | Regularization parameter to prevent overfitting                    |
| `val_size`      | `float`      | `0.15`     | Fraction of samples held out for `test_model()`.                   |


### Returns

`None` — Stores windowing settings and a trained model on the instance.

### Notes

- Call `test_model()`, `predict()`, or `save_model()` after this method.
- Rerunning `reduce_and_train()` on the same instance runs preprocessing again and continues training the same 
---

## `LightsclineCompute.test_model`

Accuracy on the internal validation split created during training.

```python
accuracy = ls.test_model()  # float in [0, 1]
```

### Returns

`float` — Fraction of validation windows classified correctly.

### Prerequisites

`reduce_and_train()` must have been called on this instance. A model loaded with `evaluate_only=True` does not have a validation split; use `predict()` instead.

---

## `LightsclineCompute.predict`

Class labels for new time-series samples using the trained (or loaded) model and the same windowing/reduction settings as training.

```python
labels = ls.predict(X_test)
```

### Parameters


| Name     | Type                                 | Description                                         |
| -------- | ------------------------------------ | --------------------------------------------------- |
| `X_test` | `list[float]` or `list[list[float]]` | 1D (one channel) or 2D (multichannel) float series. |


### Returns

One integer label **per window**. Length equals the number of windows that fit in `X_test` given `window_time` and `fs`.

### Raises

- `ValueError` — Invalid shape (not 1D/2D floats), or no model has been trained/loaded.

### Example

```python
# After reduce_and_train on multichannel training data with fs=[1700, 1200]:
one_window = ls.predict([[1.2] * 1700, [2.2] * 1200])
many_windows = ls.predict([[1.2] * (8 * 1700), [2.2] * (8 * 1200)])
```

---

## `LightsclineCompute.save_model`

Serialize the trained model and the windowing settings used at train time.

```python
ls.save_model("model.pkl")
```

### Parameters


| Name   | Type | Description                                      |
| ------ | ---- | ------------------------------------------------ |
| `path` | `str` | File path to write. Required; there is no default. |


### Returns

`True` on success.

### Raises

- `ValueError` — `path` is omitted.

### Notes

The pickle stores `window_time`, `fs`, and `per_reduction` alongside the model so `predict()` can reuse the same preprocessing after `load_model`. For the built-in network, weights are stored as a `state_dict` plus architecture (`input_len`, `hidden`, `output_len`). SAS models are stored as the fitted estimator.

---

## `LightsclineCompute.load_model`

Restore a model saved with `save_model`. Construct the instance with `evaluate_only=True`.

```python
ls = LightsclineCompute(evaluate_only=True)
ls.load_model("model.pkl", verbose=False)
```

### Parameters


| Name      | Type   | Default | Description                                              |
| --------- | ------ | ------- | -------------------------------------------------------- |
| `path`    | `str`  | —       | File path to read. Required.                             |
| `verbose` | `bool` | `False` | If `True`, prints restored windowing and architecture.   |


### Returns

`True` on success. After this call, `predict()` is available.

### Raises

- `ValueError` — `path` is omitted, or the instance was not created with `evaluate_only=True`.
- `FileNotFoundError` — `path` does not exist.

### Notes

- Do not call `reduce_and_train()` on an `evaluate_only` instance.
- `test_model()` is not available after load (there is no validation split). Use `predict()`.

---

## Complete example

```python
from lightscline.lightscline import LightsclineCompute

def make_class(label, n_samples, fs):
    return [[float(label + 0.1)] * int(fs * 10)]  # 10 seconds at fs Hz

fs = 1000
data = [
    make_class(0, 10000, fs)[0],  # class 0, one channel
    make_class(1, 10000, fs)[0],  # class 1
]

ls = LightsclineCompute(data=data, fs=fs, labels=[0, 1])

ls.reduce_and_train(
    per_reduction=90,
    window_time=0.2,
    data_aug_multiplier=2,
    layers=(10, 10),
    n_iters=500
)

print("Validation accuracy:", ls.test_model())

X_test = [0.2] * 330
print("Predictions:", ls.predict(X_test))

ls.save_model("model.pkl")

ls_loaded = LightsclineCompute(evaluate_only=True)
ls_loaded.load_model("model.pkl")
print("Loaded predictions:", ls_loaded.predict(X_test))
```

---

## Version notice

Importing `lightscline` prints:

> This is a pilot version with limited capabilities. Please contact us at [info@lightscline.com](mailto:info@lightscline.com) for the full version.

Expired License Products will not work. Contact [ayush@lightscline.com](mailto:ayush@lightscline.com) for optimization, auto-training, and other full-product capabilities.
