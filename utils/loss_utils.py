import tensorflow as tf


def running_avg_loss(loss, running_avg_loss, decay=0.999):
    """Calculates the running average of losses.

  Args:
    loss: loss of the single step.
    running_avg_loss: running average loss to be updated.
    decay: running average decay rate.

  Returns:
    Updated running_avg_loss.
  """
    if running_avg_loss == 0:
        running_avg_loss = loss
    else:
        running_avg_loss = running_avg_loss * decay + (1 - decay) * loss

    tf.logging.info('running_avg_loss: %f', running_avg_loss)
    return running_avg_loss