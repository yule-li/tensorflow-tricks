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
