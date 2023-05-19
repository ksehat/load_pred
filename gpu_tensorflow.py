import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"TensorFlow is using the following GPU(s): {gpus}")
else:
    print("TensorFlow is not using a GPU")


print("Version of Tensorflow: ", tf.__version__)
print("Cuda Availability: ", tf.test.is_built_with_cuda())
print("GPU  Availability: ", tf.test.is_gpu_available())
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))