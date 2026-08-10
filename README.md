# crystal
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Getting started

The `crystal` package provides a simple and intuitive API
for creating and manipulating high-dimensional simplexes.

![tetrahedron](images/tetrahedron.gif)

I intend to use this as an initialization method for deep learning networks.

## Information
In geometry, a simplex (plural: simplexes or simplices)
is a generalization of the notion of a triangle or tetrahedron to arbitrary dimensions.

* a 0-simplex is a point,
* a 1-simplex is a line segment,
* a 2-simplex is a triangle,
* a 3-simplex is a tetrahedron,
* a 4-simplex is a 5-cell,
* etc.

![simplexes in multiple dimensions](images/simplexes.jpg)

## Mathematics and Geometry

A regular N-simplex has N+1 vertices, each equidistant from every other vertex.

* **Vertices:** N+1
* **Edges:** $\binom{N+1}{2}$
* **Edge length:** constant $d$ between every pair of vertices
* **Centroid angle:** the angle $\theta$ between any two vertex vectors, measured from the centroid, satisfies

$$\cos\theta = -\frac{1}{N}$$

so as N grows, vertex vectors become increasingly close to orthogonal.

* **Gram matrix:** for vertex matrix $M$ (rows are vertex coordinates), the Gram matrix $MM^T$ encodes all pairwise dot products. For a regular simplex centered at the origin, this matrix has a single large eigenvalue and N equal smaller eigenvalues, a structure that makes simplexes useful as maximally-separated, symmetric initializations.
* **Volume:** the N-dimensional volume of a regular N-simplex with edge length $d$ is

$$V = \frac{d^N}{N!}\sqrt{\frac{N+1}{2^N}}$$

A full-rank simplex (edge vectors linearly independent, $\det(G) > 0$) is the
non-degenerate case. If $\det(G) \to 0$, the points have collapsed into
fewer effective dimensions than N — a degenerate simplex, regardless of
scale. "Optimal" means both full rank *and* correctly scaled; the
Eigenvalue Auto-Calibration section below handles the scale half of that.

## N-dimensional simplex creation
Creates an N-dimensional isosceles simplex centered at the origin 0.

1. First argument defines the number of dimensions.
2. Second argument defines the distance between the points.

```python
import crystal

simplex = crystal.create_simplex_matrix(2, 1.)
# simplex =
# [[ 0.14942925 -0.55767754]
#  [-0.55767754  0.14942925]
#  [ 0.40824829  0.40824829]]
```

```python
import crystal

simplex = crystal.create_simplex_matrix(5, 1.)
# simplex =
# [[ 0.5079504  -0.19915638 -0.19915638 -0.19915638 -0.19915638]
#  [-0.19915638  0.5079504  -0.19915638 -0.19915638 -0.19915638]
#  [-0.19915638 -0.19915638  0.5079504  -0.19915638 -0.19915638]
#  [-0.19915638 -0.19915638 -0.19915638  0.5079504  -0.19915638]
#  [-0.19915638 -0.19915638 -0.19915638 -0.19915638  0.5079504 ]
#  [ 0.28867513  0.28867513  0.28867513  0.28867513  0.28867513]]
```

## N-dimensional rotation matrix
Now that we have created a set of N-dimensional points defining a simplex,
we may need to manipulate them. To rotate them around the origin 0 we need
an N-dimensional rotation matrix. To create such a rotation matrix, call
`create_rotation_matrix`.

The implementation uses Givens rotations applied directly to matrix rows for
O(N³) complexity, making it efficient even for high-dimensional spaces
(tested up to 1000D).

1. An NxN matrix, where each point i,j defines a rotation in radians around that axis pair.
2. Second argument defines the cutoff in decimals and is optional.

```python
import crystal

rotation_matrix = crystal.create_rotation_matrix(
    np.array([
        [0, np.pi/4, 0],
        [0, 0, np.pi/4],
        [0, 0, 0]]),
    cutoff_decimals=5)
# rotation_matrix =
# [[ 0.70711    -0.70711     0.        ]
#  [ 0.50000455  0.50000455 -0.70711   ]
#  [ 0.50000455  0.50000455  0.70711   ]]
```

## Simplex Class

The `Simplex` class encapsulates the simplex set of points and adds functionality
for offsetting and rotating.

```python
import crystal
import numpy as np

distance = 1.
input_dims = 100
output_dims = 101
simplex = crystal.Simplex(
    input_dims=input_dims,
    output_dims=output_dims,
    distance=distance)
offsets = np.random.normal(size=(1, input_dims))
rotations = np.random.normal(size=(input_dims, input_dims))
simplex.rotate(rotations).move(offsets)
```

### `Simplex.move` allows one to offset (move) the center of the simplex.

![Simplex.move](images/add_vectors.png)

### `Simplex.rotate` allows one to rotate the simplex around the origin.

![Simplex.rotate](images/rotate.png)

### Eigenvalue Auto-Calibration

The `Simplex` class automatically calibrates the distance between points so that
the maximum eigenvalue of the scaled Gram matrix is approximately 1.0. This is critical
for numerical stability when stacking these structures in deep networks.

#### Mathematical Foundation

For a regular simplex with side length $d$ centered at the origin, the non-zero eigenvalues
of the scaled Gram matrix $\frac{1}{N}MM^T$ are approximately $\lambda \approx \frac{d^2}{2N}$.

To achieve $\lambda_{max} = 1$, the exact analytic solution is:

$$d = \sqrt{\frac{2N^2}{N+1}}$$

When `distance <= 0` is passed to the constructor, this formula is applied automatically.

#### Example

```python
import crystal
import numpy as np

input_dims = 100
output_dims = 101
simplex = crystal.Simplex(
    input_dims=input_dims,
    output_dims=output_dims,
    distance=-1)  # Auto-calibrate
m = simplex.matrix
m2 = np.matmul(m, np.transpose(m)) / input_dims
w = np.linalg.eigvals(m2)
max_eigenvalue = np.max(w)
# max_eigenvalue ≈ 1.0
```

## Deep Learning Usage (Keras)

A simplex can be used two ways in a network: as a **fixed, non-trainable**
classification head, or as a **trainable** set of vectors nudged toward
simplex geometry with a volume-based loss. Which one is right depends on
whether you know in advance that N+1 equiangular directions are the correct
target.

### Option A: Fixed simplex classifier (non-trainable)

The simplex vertices are frozen. Only the encoder producing `x` is trained;
it's pushed to align with the correct vertex. Guaranteed full rank and
correctly scaled, permanently — nothing to degenerate.

```python
import crystal
import keras
from keras import layers, ops

input_dims = 128
num_classes = input_dims + 1  # an N-dim simplex has N+1 vertices

M = crystal.create_simplex_matrix(input_dims, distance=-1)  # auto-calibrated, (N+1, N)

class SimplexHead(layers.Layer):
    def __init__(self, simplex_matrix, **kwargs):
        super().__init__(**kwargs)
        self.M = ops.convert_to_tensor(simplex_matrix, dtype="float32")

    def call(self, x):
        return ops.matmul(x, ops.transpose(self.M))  # cosine-similarity logits

inputs = keras.Input(shape=(784,))
x = layers.Dense(256, activation="relu")(inputs)
x = layers.Dense(input_dims)(x)          # encoder output, N-dimensional
logits = SimplexHead(M)(x)               # frozen simplex, N+1 outputs
outputs = layers.Softmax()(logits)

model = keras.Model(inputs, outputs)
model.compile(optimizer="adam", loss="categorical_crossentropy")
```

### Option B: Trainable vectors + volume regularization

Let representations be learned normally, and add a Gram-determinant loss
term that discourages dimensional collapse (rank deficiency), without fixing
the geometry in advance.

```python
import keras
from keras import ops

def volume_loss(vectors, eps=1e-6):
    """Encourages a batch of (N+1, N) vectors to stay full-rank / well-spread."""
    centered = vectors - ops.mean(vectors, axis=0, keepdims=True)
    gram = ops.matmul(centered, ops.transpose(centered))
    identity = ops.eye(ops.shape(gram)[0])
    sign, logdet = ops.slogdet(gram + eps * identity)
    return -logdet  # minimizing this maximizes volume

class VolumeRegularizedModel(keras.Model):
    def __init__(self, encoder, lambda_=0.01, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.lambda_ = lambda_

    def train_step(self, data):
        x, y = data
        with ops.GradientTape() as tape:
            features = self.encoder(x, training=True)
            task_loss = self.compute_loss(x, y, features)
            loss = task_loss + self.lambda_ * volume_loss(features)
        trainable_vars = self.encoder.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        return {"loss": loss, "task_loss": task_loss}
```

Use a small `lambda_` — pushing volume too aggressively fights the task loss;
too little and collapse risk returns. Track `volume_loss` as a standalone
metric during training as an early warning for representation collapse,
independent of downstream task performance.

### Which to choose

| | Fixed simplex (A) | Trainable + volume loss (B) |
|---|---|---|
| Guarantee | always full rank, always correctly scaled | encouraged, not guaranteed |
| Flexibility | geometry fixed in advance | geometry emerges from data |
| Best for | known, fixed number of classes | open-set, contrastive, self-supervised |
| Risk | wrong if N+1 classes don't match data structure | can still collapse if `lambda_` too small |
