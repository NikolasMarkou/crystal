import pytest
import crystal
import numpy as np


def test_simplex_class_10x20_offset():
    s = crystal.Simplex(
            input_dims=10,
            output_dims=20,
            distance=1.)
    offset = np.random.normal(size=(11, 1))
    s.move(offset)


def test_simplex_class_20x40_offset():
    s = crystal.Simplex(
        input_dims=20,
        output_dims=40,
        distance=1.)
    offset = np.random.normal(size=(20, 1))
    s.move(offset)


def test_simplex_class_40x20_offset():
    s = crystal.Simplex(
        input_dims=40,
        output_dims=20,
        distance=1.)


def test_simplex_class_5x10_offset_rotate():
    s = crystal.Simplex(
        input_dims=5,
        output_dims=10,
        distance=1.)


def test_simplex_class_40x20_offset_rotate():
    s = crystal.Simplex(
        input_dims=40,
        output_dims=20,
        distance=1.)


def test_simplex_class_100x200_offset_rotate():
    s = crystal.Simplex(
        input_dims=100,
        output_dims=200,
        distance=1.)

