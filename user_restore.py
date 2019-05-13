from tensorflow.python import pywrap_tensorflow

def restore_by_asign(sess,restore_vars, numpy_vars,scopes=None):
    pass

def parse_vars_by_checkpoint(checkpoint_file):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
    var_to_shape_map = reader.get_variable_to_shape_map()
    #for key in var_to_shape_map:
    #    print(key)

def user_restore(sess,restore_vars,numpy_vars,checkpoint_file,scopes=None):
    parse_vars_by_checkpoint(checkpoint_file)
    
    
def face_resave():
    '''
    Examples to show how resotre the model in which var.name is different var in tf.graph. 
    model.ckpt-2000:{'conv1/weight': array(...)}; vars in tf.graph is {'model/conv1/weight:0':array(...)}
    tf.train.Saver(var_list or var_dict): For var_dict, it use model_dict[name] which are in ckpt file to 
    initialize the var_dict[name] which is constructed by tf.graph
    '''
    from tensorflow.python import pywrap_tensorflow
    def get_tensors_in_checkpoint_file(filename,save_vars,scope_name):
        vars_dict = {}
        for v in save_vars:
            vars_dict[v.name] = v 
        reader = pywrap_tensorflow.NewCheckpointReader(filename)
        var_to_shape_map = reader.get_variable_to_shape_map()
        to_vars_dict = {}
        for key in sorted(var_to_shape_map):
            v = reader.get_tensor(key)
            from_name = '{}/{}:0'.format(scope_name,key)
            if from_name in vars_dict:
                to_vars_dict[key] = vars_dict[from_name]
        return to_vars_dict
    HEIGHT=64
    WIDTH=64
    model_path = 'models/192_tiny_area_tiny2/model.ckpt-260000'
    images = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT, WIDTH, 3], name='image_tensor')

    scope_name = 'model'
    with tf.variable_scope(scope_name):
        prelogits = face_model_forward(images,model_params)
    save_vars = [var for var in tf.global_variables() if 'global_step' not in var.name]
    print('save vars',save_vars[0])
    vars_dict = get_tensors_in_checkpoint_file(model_path,save_vars,scope_name)
    saver = tf.train.Saver(save_vars)
    print(vars_dict.keys())
    restore_saver = tf.train.Saver(vars_dict)
    sess = tf.Session()
    restore_saver.restore(sess, model_path)
    saver.save(sess,'models/tiny2_prefix/model.ckpt-260000')
