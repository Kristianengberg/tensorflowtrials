import numpy as np
import tensorflow as tf
import gym

# initialize replay memory D to capacity N
# initialize action-value function Q with random weights
# for episodes = 1, M do
    # Initialize sequence s(1) = {x(1)} and preprocessed sequence ø1 = ø1(s1)
    # for t = 1, T do
        # With probably e select a random action a(t)
        # otherwise select a(t) = max(a)Q*(ø(st),a ; Ø)
        # Set s(t+1) = s(t), a(t), x(t+1) and preprocess ø(t+1)= ø(s(t + 1)
        # Store transition (ø(t), a(t), r(t), ø(t+1) in D
        # Sample random minibatch of transitions (ø(j)a, a(j), r(j), ø(j+1) from D
        # set y(j) = r(j) for terminal _ r(j) + gamma*max(a)*Q(ø(j+1), a'; Ø) for non terminal
        # Perform a gradient descent step on (y(j) - Q(ø(j), a(j) ; Ø))^2
    #end for
#end for




n_games_per_update = 10
n_max_steps = 200
n_iterations = 10
save_iterations = 10
gamma = 0.95
learning_rate = 0.01

env = gym.make('CartPole-v0')
env.reset()


n_inputs = env.observation_space.shape[0]
n_hidden = 10
n_outputs = 1
initializer = tf.contrib.layers.variance_scaling_initializer() #look up

X = tf.placeholder(tf.float32, shape=[None, n_inputs])
hidden = tf.layers.dense(X, n_hidden, activation=tf.nn.elu, kernel_initializer=initializer)
logits = tf.layers.dense(hidden, n_outputs)
outputs = tf.nn.sigmoid(logits)  # probability of action 0 (left)
p_left_and_right = tf.concat(axis=1, values=[outputs, 1 - outputs])
action = tf.multinomial(tf.log(p_left_and_right), num_samples=1)






y = 1. - tf.to_float(action)
cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=y, logits=logits)
optimizer = tf.train.AdamOptimizer(learning_rate)
grads_and_vars = optimizer.compute_gradients(cross_entropy)
gradients = [grad for grad, variable in grads_and_vars]
gradient_placeholders = []
grads_and_vars_feed = []
for grad, variable in grads_and_vars:
    gradient_placeholder = tf.placeholder(tf.float32, shape=grad.get_shape())
    gradient_placeholders.append(gradient_placeholder)
    grads_and_vars_feed.append((gradient_placeholder, variable))
training_op = optimizer.apply_gradients(grads_and_vars_feed)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def discount_rewards(rewards, gamma):
    discounted_rewards = np.empty(len(rewards))
    cumulative_rewards = 0

    for step in reversed(range(len(rewards))):
        cumulative_rewards = rewards[step] + cumulative_rewards * gamma
        discounted_rewards[step] = cumulative_rewards
    return discounted_rewards

def discount_and_normalize_rewards(all_rewards, gamma):
    all_discounted_rewards = [discount_rewards(rewards, gamma) for rewards in all_rewards]
    flat_rewards = np.concatenate(all_discounted_rewards)
    reward_mean = flat_rewards.mean()
    reward_std = flat_rewards.std()
    print(flat_rewards)
    return [(discounted_rewards - reward_mean)/reward_std for discounted_rewards in all_discounted_rewards]




with tf.Session() as sess:
    init.run()

    for iteration in range(n_iterations):
        all_rewards = []
        all_gradients = []

        for game in range(n_games_per_update):

            current_rewards = []
            current_gradients = []

            obs = env.reset()
            for step in range(n_max_steps):

                action_val, gradients_val = sess.run([action, gradients], feed_dict={X: obs.reshape(1, n_inputs)})
                print(action_val[0][0])

            obs, reward, done, info = env.step(action_val[0][0])
            current_rewards.append(reward)
            current_gradients.append(gradients_val)
            env.render()
            if done: break

        all_rewards.append(current_rewards)
        all_gradients.append(current_gradients)

    all_rewards = discount_and_normalize_rewards(all_rewards, gamma)
    feed_dict = {}
    for var_index, grad_placeholder in enumerate(gradient_placeholders):
        mean_gradients = np.mean(
            [reward * all_gradients[game_index][step][var_index]
             for game_index, rewards in enumerate(all_rewards)
             for step, reward in enumerate(rewards)],
            axis=0)
        feed_dict[grad_placeholder] = mean_gradients
    sess.run(training_op, feed_dict=feed_dict)
    if iteration % save_iterations == 0:
        saver.save(sess, "./my_policy_net_pg.ckpt")


