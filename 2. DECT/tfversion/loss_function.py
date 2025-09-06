import tensorflow as tf
from tensorflow.keras.losses import Loss

class FocalLoss(Loss):
    def __init__(self, alpha=1.5, gamma=2.0):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def call(self, y_true, y_pred):
        # Calculate focal loss for multi-class classification with one-hot encoded labels
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.clip_by_value(y_pred, 1e-10, 1.0 - 1e-10)

        alpha_factor = tf.ones_like(y_true) * self.alpha
        alpha_factor = tf.where(tf.equal(y_true, 1), alpha_factor, 1 - alpha_factor)

        focal_weight = tf.where(tf.equal(y_true, 1), 1 - y_pred, y_pred)
        focal_weight = self.alpha * tf.pow(focal_weight, self.gamma)

        focal_loss = -tf.reduce_sum(alpha_factor * y_true * tf.math.log(y_pred) * focal_weight, axis=-1)

        return focal_loss

def calculate_virtualsoftmax_logits(inputs, labels, num_classes, mode, dtype=tf.float32):
    embedding_size = inputs.get_shape().as_list()[-1]
    kernel = tf.compat.v1.get_variable(name='final_dense', dtype=tf.float32, shape=[embedding_size, num_classes], 
                                       initializer=tf.keras.initializers.glorot_normal, 
                                       trainable=True)
    if dtype == tf.float16:
        kernel = tf.cast(kernel, tf.float16)
    # calculate normal WX(output of final FC)
    WX = tf.matmul(inputs, kernel, name='final_dense_logits')

    if mode == tf.estimator.ModeKeys.TRAIN:
        # get label indices to get W_yi
        W_yi = tf.gather(kernel, labels, name='W_yi', axis=1)
        W_yi_norm = tf.norm(W_yi, axis=0, name='W_yi_l2norm')
        W_yi_norm = tf.transpose(W_yi_norm, name='W_yi_l2norm_transpose')
        X_i_norm = tf.norm(inputs, axis=1, name='X_i_l2norm')
        # calculate WX_virt => virtual class output
        WX_virt = tf.multiply(W_yi_norm, X_i_norm)
        WX_virt = tf.clip_by_value(WX_virt, 1e-10, 15.0) # for numerical stability
        WX_virt = tf.expand_dims(WX_virt, axis=1)
        # new WX is normal WX + WX_virt (concatenated to the feature dimension)
        WX_new = tf.concat([WX, WX_virt], axis=1, name='vsoftmax_logits')
        return WX_new
    else:
        return WX