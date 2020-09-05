# crystal

In geometry, a simplex (plural: simplexes or simplices) 
is a generalization of the notion of a triangle or tetrahedron to arbitrary dimensions.

* a 0-simplex is a point,
* a 1-simplex is a line segment,
* a 2-simplex is a triangle,
* a 3-simplex is a tetrahedron,
* a 4-simplex is a 5-cell.
* etc ...  

## N-dimensional simplex creation
Creates an N-dimensional isosceles simplex that has it's center at the origin 0.

1. First argument defines the number of dimensions.
2. Second argument defines the distance between the point 

```python
simplex = crystal.create_simplex(5, 1.)
```

##  N-dimensional rotation matrix
To manipulate N-dimensional points we need a N-dimensional rotation matrix
```python
simplex.create_simplex(
```