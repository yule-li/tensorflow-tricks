This project summaries many useful tricks to tackle some difficult problems.


## static shape and dynamic shape
[Reshape the tensor with some dimension as ```None```.](https://github.com/yule-li/tensorflow-tricks/blob/master/reshape_with_none.ipynb)


## save and restore
Sometimes, you want to initialize the different parts using the same checkpoints. For example, you have a network with two branches that have the same structure, and you want to initialize the both branch using the same checkpoints. We can achieve this using the ```checkpoint_utils``` as bellow:

```
from tensorflow.python.training import checkpoint_utils
...
with tf.variable_scope('left_branch) as src:
    left_logits,_=resnet_v2_50(images,...)
with tf.variable_scope('right_branch) as src:
    right_logits,_=resnet_v2_50(images,...)
#init the two branch network from the same checkpoint.
checkpoint_utils.init_from_checkpoint(checkpoint_file,{'/':'left_branch'})
checkpoint_utils.init_from_checkpoint(checkpoint_file,{'/':'right_branch'})
# using {'name':var}
#checkpoint_utils.init_from_checkpoint(checkpoint_file,name_var_pair)
```

The resotre operation just assign the variable using the value from checkpoint, so a more flexible method to do this is to do assign by ourself as shown bellow:
```
def user_restore(sess,checkpoint_file,restore_var):
    reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_file)
    assign_vars = []
    for name in restore_var.keys():
        value = reader.get_tensor(name)
        assign_vars.append(tf.assign(restore_var[name],value))
    assign_group = tf.group(*assign_vars)
    sess.run(assign_group)
```

##  Graph editor
Tensorflow program includes two phases:1)assemble the graph;2)run the operation defined in graph in session. So the graph is static and we can obtains all informations about the graph and can also modify the graph. For example, given a tensor we can get its parent nodes and children nodes. Further, if needed we can change the link between  a node and b node.

### [change the connection of nodes](https://github.com/yule-li/tensorflow-tricks/blob/master/graph_connect/node-merge.ipynb)
We change the link that a->b and c->d to a->b and c->b.
```
import tensorflow.contrib.graph_editor as ge
ge.reroute_ts(b,d)
```
