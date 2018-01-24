import tensorflow as tf

from datasets import dataset_factory
from preprocessing import preprocessing_factory
from nets import lenet

slim = tf.contrib.slim

model = '/runtmp2/lynden/tensorflowCodes/TensorQuant/slim/train_log/lenet/model.ckpt-120000.meta'
train_dir = '/runtmp2/lynden/tensorflowCodes/TensorQuant/slim/train_log/lenet'
data_path = '/runtmp3/lynden/imageDataset/mnist/'
batch_size = 10


dataset = dataset_factory.get_dataset(
        'mnist', 'test', data_path)

provider = slim.dataset_data_provider.DatasetDataProvider(
        dataset,
        shuffle=False,
        common_queue_capacity=8 * batch_size,
        common_queue_min=batch_size*4)

[image, label] = provider.get(['image', 'label'])
		
# select the preprocessing function
preprocessing_name = 'lenet'
image_preprocessing_fn = preprocessing_factory.get_preprocessing(
        preprocessing_name,
        is_training=False)
eval_image_size = 28
image = image_preprocessing_fn(image, eval_image_size, eval_image_size)


sess = tf.Session()
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
sess.run(init_op)

images, labels = tf.train.batch(
        [image, label],
        batch_size=batch_size,
        num_threads=1,
        capacity=5 * batch_size)

coord = tf.train.Coordinator()
threads = tf.train.start_queue_runners(sess=sess, coord=coord)
#img, lab = sess.run([images, labels])

#print 'first batch:'
#print 'lab:', lab
#print 'img: ', img

#images = tf.cast(images, tf.float32)

logits, endpoints = lenet.lenet(images)
# convert prediction values for each class into single class prediction
predictions = tf.to_int64(tf.argmax(logits, 1))
labels = tf.squeeze(labels)

saver = tf.train.Saver()
saver.restore(sess, tf.train.latest_checkpoint(train_dir))

preResult = sess.run([predictions, labels])
print preResult
#print sess.run(labels)


# Define the metrics:
#names_to_values, names_to_updates = slim.metrics.aggregate_metric_map({
#    'Accuracy': slim.metrics.streaming_accuracy(predictions, labels),
#    'Recall_5': slim.metrics.streaming_recall_at_k(logits, labels, 5),
#})

#sess.run(tf.global_variables_initializer())
#sess.run(tf.local_variables_initializer())
#metric_values = sess.run(names_to_values.values())
#for metric, value in zip(names_to_values.keys(), metric_values):
#    print('Metric %s has value: %f' % (metric, value))
#print sess.run(names_to_values.values())

# Stop the threads
coord.request_stop()
# Wait for threads to stop
coord.join(threads)

sess.close()