# coding: utf-8
import tensorflow as tf

def smooth_l1_loss(y_true, y_pred, use_mean=True):
	delta = 0.5
	x = tf.abs(y_true - y_pred)
	xx = tf.where(x < delta, 0.5 * x ** 2, delta * (x - 0.5 * delta))
	return tf.reduce_mean(xx) if use_mean else tf.reduce_sum(xx)