import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

# Create a simple TensorFlow computation graph
with tf.device('/device:GPU:0'):
    a = tf.constant(5.0, shape=[2, 3])
    b = tf.constant(10.0, shape=[3, 2])
    c = tf.matmul(a, b)

# Run the computation graph and print the result
result = c.numpy()
print(result)
