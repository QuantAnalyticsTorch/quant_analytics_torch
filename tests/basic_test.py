import tensorflow as tf

EPSILON = 0.0001

def test_zero():

    assert abs(0.0) < EPSILON


def test_another_zero():
    
    assert abs(0.0) < EPSILON

if __name__ == '__main__':
    
    test_zero()