import tensorflow as tf

input1 = tf.placeholder(tf.float32)
input2 = tf.placeholder(tf.float32)
output = tf.mul(input1, input2)

with tf.Session() as sess:
    print(sess.run([output], feed_dict={input1:[7.], input2:[2.]}))

# 输出:
# [array([ 14.], dtype=float32)]
#
# a = tf.placeholder(tf.type.float32)
# b = tf.placeholder(tf.type.float32)
#
# c = tf.add(a, b)
# d = tf.mul(a, b)
#
# with tf.Session() as sess:
#     print(sess.run([c, d], feed_dict={a: [7.], b: [8.]}))
#
