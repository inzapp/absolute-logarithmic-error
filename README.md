# Absolute Logarithmic Error

ALE(Absolute Logarithmic Error) is an improved function of the BCE(Binary Cross Entropy) that can be used for both classification and regression problems

ALE can calculate almost the same gradient that BCE intends with fewer operations than BCE

And it completely solves the problem of solving the regression problems that BCE has

I think ALE can replace BCE perfectly

First, let's look at the BCE formula below

<img src="https://user-images.githubusercontent.com/43339281/197669369-6af8d296-457b-41bc-8de1-eba1131bd51f.png" width="600px">

<img src="https://user-images.githubusercontent.com/43339281/197914245-5a51b491-04c3-4d64-846c-debbb29a0d0a.png" width="400px">

Sum two values after calculation for label 1 and 0 respectively

The formula basically assumes that label is 1 or 0

This has a fundamental problem

The problem is that the loss value for a not 0 or 1 label is invalid

The value of BCE loss if label is 1 or 0,

```python
>>> loss_fn = tf.keras.losses.BinaryCrossentropy()
>>> 
>>> y_true, y_pred = [[1.0]], [[0.0]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 15.424949
>>> 
>>> y_true, y_pred = [[1.0]], [[0.2]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 1.6094373
>>> 
>>> y_true, y_pred = [[1.0]], [[0.9]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.10536041
>>> 
>>> y_true, y_pred = [[1.0]], [[1.0]]
>>> print(loss_fn(y_pred, y_pred).numpy())
>>> 0.0
```

The closer you get to the target value, the lower loss is returned

This does not seem to be a problem. However, if the label is a value between 0 and 1,

```python
>>> loss_fn = tf.keras.losses.BinaryCrossentropy()
>>> 
>>> y_true, y_pred = [[0.1]], [[0.1]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.32508278
>>> 
>>> y_true, y_pred = [[0.2]], [[0.2]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.50040215
>>> 
>>> y_true, y_pred = [[0.5]], [[0.5]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.69314694
>>> 
>>> y_true, y_pred = [[0.6]], [[0.6]]
>>> print(loss_fn(y_pred, y_pred).numpy())
>>> 0.6730114
```

Although the model predicts a value that is exactly the same as the target value, each returns a different loss

This can cause instability when training the regression model

And also means that loss value cannot be used as an appropriate metric when solving the regression problem using BCE

If the error value of the predicted value is equal to the target value, the loss function must return the same value regardless of the target value

ALE provides the correct loss value and gradient even when training using continuous labels between 0 and 1

<img src="https://user-images.githubusercontent.com/43339281/201013651-fea34060-8f20-42a5-b480-284dc183ee59.png" width="400px">

<img src="https://user-images.githubusercontent.com/43339281/197663446-3e3aa5dc-a332-417e-9bfa-3df38b7c7bf2.png" width="400px">

It also has fewer operations, more intuitive formulas, and easier to optimize than BCE

And ALE doesn't have a problem of returning a different loss even if it has the same error according to the target value, which is a problem of BCE

```python
>>> loss_fn = AbsoluteLogarithmicError()
>>> 
>>> y_true, y_pred = [[1.0]], [[0.0]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 15.249238
>>> 
>>> y_true, y_pred = [[1.0]], [[0.2]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 1.6094373
>>> 
>>> y_true, y_pred = [[1.0]], [[0.9]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.10536041
>>> 
>>> y_true, y_pred = [[1.0]], [[1.0]]
>>> print(loss_fn(y_pred, y_pred).numpy())
>>> 0.0
>>> 
>>> y_true, y_pred = [[0.0]], [[0.0]]
>>> print(loss_fn(y_pred, y_pred).numpy())
>>> 0.0
>>>
>>> y_true, y_pred = [[0.1]], [[0.1]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.0
>>> 
>>> y_true, y_pred = [[0.2]], [[0.2]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.0
>>> 
>>> y_true, y_pred = [[0.5]], [[0.5]]
>>> print(loss_fn(y_true, y_pred).numpy())
>>> 0.0
>>> 
>>> y_true, y_pred = [[0.6]], [[0.6]]
>>> print(loss_fn(y_pred, y_pred).numpy())
>>> 0.0
```

## Compare to Focal Loss

ALE can also be compared to Focal loss

Focal loss helps focus on hard samples by giving lower weights to easy samples that BCE can classify

ALE can also do this the same and it's very simple and intuitive

<img src="https://user-images.githubusercontent.com/43339281/201014039-cc008328-24ee-435d-b9d6-b86058079a78.png" width="600px">

<img src="https://user-images.githubusercontent.com/43339281/197667902-363e096f-f43a-40ff-9b62-ff5e54b867a0.png" width="400px">

The table below shows the results of training using the cifar10 dataset for comparison between BCE and ALE

The model was trained with the same hyperparameters and tested only by changing the loss function

Model params : 4,698,186

batch size : 128

lr : 0.003

momentum : 0.9

epochs : 10 (Adam for 7 epochs, Nesterov SGD for 3 epochs)

| Gamma | BCE | ALE |
|-|-|-|
0.0	| **0.8631** | 0.8628
1.0	| 0.8473 | **0.8584**
2.0	| 0.8488 | **0.8574**
avg	| 0.8531 | **0.8595**
