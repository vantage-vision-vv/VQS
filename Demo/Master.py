import os
import time
import tensorflow as tf

# read files 
demo_dir = 'Demo/trim_videos/'
files = os.listdir(demo_dir)

start = time.time()
# action classification
with tf.Session() as sess:
    # load saved model
    saver = tf.train.import_meta_graph('Models/VIRAT/videolstm_model-42.meta')
    saver.restore(sess, tf.train.latest_checkpoint('Models/VIRAT/'))
    
    graph = tf.get_default_graph()
    
    Z = graph.get_tensor_by_name('input:0')
    out = graph.get_tensor_by_name('output:0')
    Att_t = graph.get_tensor_by_name('attention:0')
    
