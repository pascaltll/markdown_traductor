# Content

Book of machine learning

(images/d65809172d0d826cfd5df6d9ac9c02569ce45091ea65965c8e1edfed5dd273c4.jpg)

11.1 Training with support

The authors

(images/32d40fbe70e1e4e4ab254a7a2aac15dfce151fb2bd65de2e54f43daff379842e.jpg)

Ivanov Sergey

So far, the experience that made it possible to learn in our algorithms has been given in the form of a learning sample. How much such a learning model relates to the way one learns, for example, a person? To learn to ride a bicycle, cook a cake or play tennis, we don’t need huge dataset with examples of what to do at any time; instead we are able to learn by trial and error method, making attempts to solve a task, interacting with the surrounding world, and somehow improving our behavior based on the experience obtained during this interaction.

In reinforcement learning (RL) we want to build an algorithm modeling learning by the method of samples and errors. Instead of receiving a learning sample to the entrance such an algorithm will interact with some environment (environment), the surrounding world, and in the role of a "marking" will be a reward (reward) - a scalary value that is issued after each step of interaction with the environment and shows how well the algorithm deals with the task placed on it. For example, if you bak a cake, then for each cake you get $+1$, and if you try to ride a bicycle, then for each fall of the bicycle you get a 1.

The prize does not indicate how you need to solve a task and what you need to do at all; the prize can be delayed in time (you have found a treasure in the desert, but to get the deserted cakes, you will still need a lot of time to get out of the desert; and the prize comes only for the cakes) or strongly cut (most of the time to give an agent $+0$ ).

(images/d0f474ca25be1c278d43ecb12188a68e50c212944d13669312803aa6ccb1b0df.jpg) image source — UC Berkeley AI course

Setting a task

The task of learning with support is set by the Markov Decision Process (MDP) is four $\left(\boldsymbol{\mathcal{S}},\boldsymbol{\mathcal{A}},\boldsymbol{\mathcal{P}},\boldsymbol{\r} \right)$, where:

A is an action space, a variety of options from which you have to make a choice at each step of your interaction with the environment. $\mathcal{P}$ is a transition function that determines the change of the environment after in $s\in$ $s$ has been selected action $a\in$ $\mathcal{A}$ In general, the transition function can be stochastic, and then such a transition function is modeled by the distribution of $p(s^{prime}\mids,a)$: with what probability the environment will be in a state after the choice of action $a$s in $s$s.

$r{:}S\times$ $A\rightarrow\mathbb{R}$ is a reward function that gives a scalary value for the choice of action $a$ in the state of s. This is our "learning signal".

The agent is guided by some rule, also stochistic, how to choose actions depending on the current state of the environment, which is called a strategy (policy; the term is often translated and said policy) and is modeled by the distribution of $\pi(a♰\quad s)$. Strategy and will be our search object, so as in classical machine learning, we are looking for some function.

The agent simulates the action from its strategy of transitions $a_{0}\sim$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

(images/a70f06c857f6229545ccb1c99a6a8f9d583b743c9e041c9a236caaccdb2a2564.jpg)

So, the $\Phi$ active environment for us is a controlled mark chain: at each step we choose $a$ to determine the distribution from which the following state will be generated. We assume, first of all, the mark property: that the transition to the following state is determined only by the current state and does not depend on the entire previous history:

$$ p(s_{t+1}\mid s_{t},a_{t-1},a_{t-1},\ldots,s_{0},a_{0})=p(s_{t+1}\mid s_{t},a_{t}) $$

Second, we assume stagnation: the function of transitions $p(s^{\prime}\mid s,a)$ does not depend on time, how many steps have passed since the beginning of interaction. These are quite realistic assumptions: the laws of the world do not change over time (stagnation), and the state - describes the world as a whole (marking). In this interaction model there is only one unrealistic assumption: full observability (full observability), which says that the agent in its strategy $\pi(a♰\quad s)$ observes the whole state s completely and can choose actions, knowing about the surrounding world absolutely everything; in reality we have only some part observations of the state.

So we learned in mathematical language to model the environment, the agent and their interaction with each other. It remains to understand what we want. During interaction at each step the agent comes a reward of $r_{t}=r(s_{t},a_{t})$, however, the state and action of $s_{t}$, $a_{t}$ within such a layout are random values. The same agent can by the force of the stochastics both internal (by the force of the choice of action in its strategy), and external (by the force of the stochastics in the function of transition) to obtain a very different total reward of $\textstyle\sum_{t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t\t

Every $\pi$ strategy sets a distribution in the trajectory space — with what probability we can encounter the trajectory $\tau=$ $\left(s_{0},a_{0},s_{1},a_{1},\dots\right)$ :

$$ p({\mathcal T} p({\mathcal T} p({\textit{\pi}})=p(s_{0},a_{0},s_{1},a_{1},\cdot\cdot\cdot<{\textit{\pi}})=\prod_{t\geq0}p(s_{t+1} <s_{t},a_{t}){\pi}(a_{t} <s_{t}) $$

This is according to this distribution we want to take the average of the agent receiving awards. They usually record it somehow like this:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Here mat. expectation on the trajectory is an infinite chain of mat. expectations:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

In environments where interaction can last indefinitely long, the agent can learn to get an infinite reward, with which different paradoxes can be associated (for example, getting $+1$ on each second step becomes also good as getting $+1$ on each hundred steps). Therefore, they introduce discounting (discounting) rewards, which says: the cake is now better than the same

The reward we will receive in the future, the agent will discount to a certain number of $\gamma$, a smaller unit. Then our functionality will take the following kind:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

(images/ee89d39c3619039aeb8b6ee15ea83fc198582335fd617f049e7bd8a03a9ccda6.jpg) image source — UC Berkeley AI course

Let’s note that training with support is primarily the task of optimizing, optimizing functions of a particular type. If in classical machine learning the selection of the loss function can be considered an element of the engineering part of the solution, then here the reward function is assigned to us ready and determines the function that we want to optimize.

# Examples

The formalism of the MDP is very common, and it falls under almost everything that can be called a “intellectual task” (with the provision that it is not always clear which function of reward puts one or another task).

For example, the most simple examples of MDP can be drawn "on paper." For example, they are often considered "celled worlds" (GridWorlds): an agent is in some position of a fiber plate and can as an action choose one of the four directions. Such worlds can respond differently to the agent for the choice of action "to go to the wall", with a certain likelihood of moving the agent not in the direction he chose, containing objects in some cells and so on.

(images/1e062fa68024bb497fa5eef0a680e2f63669bf82f497dae426959e29ecb57a37.jpg) image source — UC Berkeley AI course

A huge variety of MDP provides video games. You can think that the entrance agent will be submitted a picture of the video game screen, and several times a second the agent chooses which buttons on the controller he wants to press. Then the space of states - a lot of all kinds of images that can show you the video game. Multiple, in general, final (the final number of pixels of the screen with three color channels, each of which shows a total value from 0 to 255), but only very large; for example, they can no longer be listed or preserved all possible options in memory. But at each step you need to choose the action from the final to: which buttons to press, so this task is discreted management.

(images/e590a4f7966ec2627b3c0037d613d81d0f1351b4d40707d6905679ab73a9da5a.jpg)

Image source: UC Berkeley AI

Finally, the natural way to create the environment is the use of physical simulations. As a benchmark, locomotion is often used - the task of teaching any "existence" to walk within the framework of a physical model (examples can be seen, for example, here). Conceptually, within the framework of the task of learning with strengthening, we don't even matter how the simulation is organized or how the reward function is set: we want to build a common algorithm of optimization of this same reward. If the reward encourages the movement of the center of the mass "existence" along a certain direction, the agent will gradually learn to choose actions so that the existence is moving and not falling, if later it leads to the end of the episode and receives further reward.

(images/5d0bf19a3fb6829475d535735516357593e60071be9780215b38ffad49dc588a.jpg) image source — article DeepMind Producing Flexible Behaviours in Simulated Environments

In such tasks, the agent at each step chooses several material numbers in the range [−1,1 ] where -1 is "maximally relax" the joint, and $+1$ is "maximally tension." Such an action space occurs in many tasks of robotics, where you need to learn to turn some roll, and it has the extreme right and the extreme left position, but you can choose any intermediate. Such tasks are called the tasks of continuous control.

OK, and how to decide?

It seems complicated, but mankind has already a lot of work on how to approach this as a very common task, with the basic idea you’ve probably already encountered.

The fact is that we optimize not to any functionality, but the average discounted cumulative reward. To invent a more effective solution than any approach that does not use this fact (e.g., evolutionary algorithms), we need to take advantage of the structure of the assigned task. This structure is set in the formalism of the MDP and the definition of the process of interaction of the agent with the environment. Intuitively it is expressed as follows: here we sit in some state of s and we want to choose the action $a$ as optimal as possible. We know that after choosing this action we will receive a reward for this step $r=r(s,a)$, the environment will move us into the state of s and the attention, we will continue to take advantage of the structure of the process: this structure is set in the formalism of the MDP and the process of interaction of the agent with the environment.

To formulate this in the language of mathematics, the “additional variables” are introduced, auxiliary values called assessment functions. Let’s get acquainted with one such assessment function – the optimal Q function that we’ll mark $Q^{*}(s,a)$. Let’s say that $Q^{*}(s,a)$ is the amount of maximum rewards you can (in average) obtain after choosing the action of $a$ from the state of s. So:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

The recording of ${\mathcal{T}}\sim\pi♰~s_{0}=s,a_{0}=a$ here means that we are sitting in the state of $s_{0}=s$; we select the action of $a_{0}=a$, and then we continue to interact with the environment using the strategy of $\pi$, thus generating the trajectory of T. In order to calculate $Q^{*}(s,a)$, you need to recover all the strategies, see how much each of them receives rewards after choosing $a$ from the state of s, and take the best strategy. Therefore, this assessment function is called the optimal: it implies that in the future after choosing the action of $a$ from the state of $s agent will behave optimally.

Definition is unconstructive, of course, because in reality we cannot do so, but it has an interesting property. If we somewhat miraculously learned $Q^{*}(s,a)$, then we know the optimal strategy. In fact: imagine that you are in the state of $s$, you have to make a choice of three actions, and you know the values of $Q^{*}(s,a)$. You know that if you choose the first action $a=0$, then in the future you will be able to obtain no more than, supposedly, $Q^{*}(s,a=0)=3$ rewards. You know that there is a strategy of $pi$ on which the maximum of the optimum Q-function is achieved, so that it allows you to do a choice of three actions, and you know that if you choose the first action of $a=1$, then you will be able to obtain a choice of more than $1$$, then you will be able to obtain a choice of $1$, and you will be able to obtain a choice of more than $1$.

The choice of the action at which the maximum is achieved by the actions of the Q function is called greedy (greedy) in relation to it. Thus, the principle of optimality of Bellman says:

The desired choice for the optimal Q function is optimal:

$$$$$$$$$$$$$$$$$$$$$$$$

Note: If the Q function reaches the maximum on several actions, you can choose any of them.

Let’s note that this optimal strategy is determined. This interesting fact means that we generally don’t need to look for a stochistic strategy. Our consideration so far even shows that we can simply try to find $Q^{*}(s,a)$ and then get out of it the optimal strategy by choosing an action thirsty.

But how to find the following $Q^{*}(s,a)$? here on the scene and appears our observation of the task structure. It appears that $Q^{*}(s,a)$ is expressed by itself. Indeed: let's consider a pair of state-action $s,a$. On the one hand, by definition, we will be able in the future under the conditions of optimal behavior to get $Q^{*}(s,a)$ rewards. On the other hand, after we choose the action $a$ in the state of $s, we will receive a reward for one step $\boldsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbolsymbols

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

We have a system of equations that connects the values of $Q^{*}(s,a)$ with itself. This is a nonlinear system of equations, but it turns out that it is in some sense "good." It has the only solution - and thus, the solution of this equation can be considered an equivalent definition of $Q^{*}(s,a)$ - and it can be sought by the method of simple iteration. The method of simple iteration of the solution of the equation systems allows to improve its current approximation of $x$ to the solution of some equation of the type $x=f(x)$ by setting it in the right part. That is, we will initialize the arbitrary function of $#######################

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

By the way, when you have met in the past with dynamic programming, you are likely to have used this idea unclearly, except that often in the tasks to solve the equations of optimality of Bellman you can simply consistently exclude the unknown variables; but the method of simple iteration gives a more general scheme, always applicable. And now for us the following principle: if we have some approximation of $Q^{*}$, then the calculation of the right part of the equation of optimality of Bellman will allow a better approximation.

Where are the methods of trials and errors?

To solve the method of simple iteration of the Bellman optimity equation and thus get $Q^{*}(s,a)$ in reality can only be under two very substantial restrictive conditions. It is necessary that, first, we can store some current approximation of $Q_{k}^{*}(s,a){bf\delta B}$ memory. This is possible only if the states and actions spaces are final and not very large, that is, for example, in your MDP only 10 states and 5 actions, then $Q^{s,a)-mathsf{s T o} table 10x5. But what if you want to learn to play the video and this state is in the image then a lot of cards that can show you the state and action spaces are final and not very large, that is, for example, in your MDP only 10 states and 5 actions, then $Q^{s,a)-mathsf{s T o} table 10x5.

Secondly, we need to be able to count the expression that stands to the right in the Bellman optimity equation:

$$ r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime} a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}) $$

A little that in complex environments take mat. expectation by the transition function $\mathbb{E}_{s^{\prime}\sim p\left(s^{\prime}\mid s,a\right)}$ in reality we cannot, so still and usually we do not really know this transition function. Imagine that you ride a bicycle: can you by the current state of the surrounding world, for example, the position of all the atoms in the universe, tell us with what chances in what state the world will be at the next moment? This consideration also suggests that it would be good if we could solve the task, avoiding even trying to find this complex transition function.

What is available to us? We can take any $\pi$ strategy (important point: we must choose one) and interact it with the environment. “Test to solve a task.” We can generate using $\pi$ a whole trajectory or even make just one step in the environment. Thus we collect data: we may be in a state of s and have made a choice of action only $\a$, then we will learn what reward we will receive for such a step and, most importantly, what state of $s^{prime} we have translated the environment. We can generate a full trajectory or even make a single step in the environment. Thus we will collect data: we may be in a state of s and have made a choice of action only $\a$\a$\a$\a$\a$\a$\a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a$a

$$ r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime} a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}), $$

which we cannot count, and on his Monte Carlo assessment:

$$ r(s,a)+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}), $$

Where is $s^{\prime}$ a sample of the transition function from the experience we have collected? In average, such a replacement is true. Such a Monte Carlo rating of the right part for the given transition $\left(s,a,r,s^{\prime}\right)$ is called the Belman target, i.e. the "target variable." Why such a name - we will see a little later.

Let us admit that after performing the action of $a$ from a state of $s$ the medium rewards us $r(s,a)=0$ and replace us with equal probability to the state of $s^{\prime}$ for which $\begin{array}{r}{operatorname} as well as ${max}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}{r}}{r}{r}{r}{r}}{r}{r}{r}{r}{r}}{r}}}{r}}}}}}{r}}}{r}}}{r}}{r}{r}{r}}}}}}{r}}}{r}{r}{r}{r}{r}}}}}{r}}}}}}}}{r}}{r}}{r}}}}

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Thus, we carry out an exponential smoothing of the old approximation of $Q_{k}^{*}(s,a)$ and a new assessment of the right part of the Bellman optimity equation with a fresh sample of $s^{\prime}$. The choice of $\alpha$ here determines how much we pay attention to the last sample, and has the same physical meaning as the learning rate.

$$ r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime} a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}), $$

And that means to apply the "smalled" method of simple iteration.

We will somehow interact with the environment and collect passages $\left(s,a,r,s^{\prime}\right)$. For each passage we will update one cell of our Q-table size number of states to number of actions according to the above formula. Thus we will get as a "smalled" method of simple iteration, where we at every step update only one cell of the table, and do not replace a hard value to the right part of the equations

optimality, but we just move in some in the average correct stochastic direction.

It seems that such an algorithm coincides with the real $Q^{*}(s,a)$ if for any pair of $s,a$ we do an infinite number of updates throughout the process, and the learning rate (hyperparameter of $\alpha$) in them behaves as the learning rate from the conditions of the coincidence of the stochistic gradient drop:

$$ \sum_{i}\alpha_{i}=+\infty,\qquad\sum_{i}\alpha_{i}^{2}<+\infty $$

The example

Let's admit, in some state s he performed the action $a$ "buy a ticket" and suggests that in the future he will be able to obtain the current rating of the next prize in case of refusal to buy a ticket equal to $0$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^ ^

Suppose, learning rate $\alpha=0.5$ : then, moving +100 toward $+90$, expectations from the future prize after buying a lottery ticket fall to 95. Everything else $+95>0$, so the column seems that buying a ticket is more profitable than not buying, so let's consider the next transition. Suppose, the column again bought a ticket, again lost $10 and again got in the same $s^{\prime}=s.$. Our update again will say to reduce the value of $Q^{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}

(images/46ad0328877815b26d19342967c906411a503ccb5937922bbeeb87b0a2057efc.jpg)

It seems that if the column continues so hard, the target will be 10 times less than the current approximation, and $Q^{*}(s,a)$ will all decrease and decrease until it falls to zero (and there will be more profitable not to buy a ticket). But if on the next iteration the Column is lucky, and the medium has translated it into $s^{\prime}$, the corresponding victory in the lottery (and this is apparently happening with some small probability), the target will become very big, and the approximation of $Q^{*}(s,a)$ our update will say to increase significantly:

[ ](images/d54fdd3121ba7c58b8c1d2b9c85dbcb3a1c63370080eb125e0e17592592357b8.jpg)

Let’s assume that the medium for the purchase of a lottery ticket corresponds with the likelihood of $p$ return to the same state of $s^{\prime}=s$ where the bulk is offered to buy another ticket, and with the likelihood of $p$ the ticket becomes profitable, and the bulk falls into such state of $s^{\prime}$ where it can take the prize and get $+1000$ (after that interaction with the medium, let’s say, ends). Let’s record the Bellman’s optimum equation for the action of $a$ “buy a ticket” in the state of s:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Here is $\operatorname*{max}_{a^{\prime}}$$Q^{*}(s^{\prime}=s,a^{\prime})=\operatorname*{max}(Q^{*}(s,a),0)$ as a cowboy can either buy a ticket or not buy (this, suppose, will bring him 0 rewards).

$$ Q^{*}(s,a)=-10+p\mathrm{max}\left(Q^{*}(s,a),0\right)+1000(1-p) $$

It is clear that if the likelihood of losing in the lottery is $p=0.99$, then the solution of the equation is $Q^{*}(s,a)=0.$ : Colabulka pays for the ticket $10 and receives 1000 rewards with the likelihood of 0.01. In this case the action "buy the ticket" and "not buy" are equal, and both in the future will bring an average of 0 rewards. If $p>0.99$, then buy the ticket becomes unprofitable, and if $p<0.99$, then it is profitable to buy the ticket until the victory occurs.

This algorithm, to which we have already practically come, is called Q-learning, “learning of optimal Q-function.” However, we still have to answer one question: how do we need to collect data to meet the requirements for compatibility? How do we interact with the environment so that we each cell s,and not cease to update?

The dilemma of exploration-exploitation

The task of multi-hand bandits that met there is actually a private case of the training task with strength, in which after the first choice of action the episode is guaranteed to be completed, and this private case of the task is often used to study this dilemma. Let's consider this dilemma in our context.

Suppose, on the next step of the algorithm we have some approximation $Q_{k}(s,a)\approx$ $Q^{*}(s,a)$. This approximation is of course inaccurate, since the algorithm, if it falls to the real optimal Q-function, then on infinity. How do you need to interact with the environment? If you want to get the maximum reward, you probably deserve to take advantage of our theory and engage in exploitation by choosing an action thirsty:

$$ \pi(s)=\operatorname{argmax}_{a}Q_{k}(s,a) $

Unfortunately, such a choice is not a fact that coincides with the real optimal strategy, and the most important thing, it is determined. This means that when this strategy interacts with the environment, many couples s,a will never meet simply because we never choose the action $a$ in the state s. And then we, it turns out, risk never again to update the cell $Q_{k}(s,a)$ for such couples!

We wanted to learn to ride a bicycle and got $+0.1$ for each passed meter and -5 for each falling into a tree. After the first trials and errors we found that riding a bicycle brings us -5, as we very soon crashed into a tree and updated our approximation of Q Function samples with a negative reward; so if we don’t even riding a bicycle and simply take a bicycle we don’t do anything, then we will be able to avoid trees and get 0. Just because in our strategy of interaction with the environment we never met $s, which lead to a positive reward, and our current strategy of approximation with a negative reward; so if we don’t even riding a bicycle and simply take a bicycle we don’t do anything, we will get 0.

The exploration mode implies that we interact with the environment using any stochistic strategy $\forall s,a\colon\pi(a♰\quad s)>0.$ For example, such a strategy is a random strategy choosing random actions. How strange, the collection of experience using random strategy allows you to live with a non-zero probability in all areas of the state space, and theoretically even our Q-function learning algorithm will match. Does this mean that the exploration is enough, and the exploitation can be hit?

In reality, we understand that to get to the most interesting areas of space state, where the reward function is the greatest, not so simple, and the random strategy although it will do this with a non-zero probability, but the probability that will be exponentially small. And for consistency we need to update the cells of $Q_{k}(s,a)$ for these interesting states endlessly many times, that is, we will have to wait for an unusually rare connection far from a time. Where it is more reasonable to use the already existing knowledge and with the help of a thirsty strategy that already knows something to go to these interesting states. Therefore, to solve the dilemma exploration-exploitation will usually take our current desire strategy and what it does with it, so that it becomes a little bit of a chance, that we will have to wait for an unusually rare connection far from a time. where it is more reasonable to use the already existing knowledge and with a thirsty strategy that already knows something.

Let’s assert what we have achieved, in the form of a table learning algorithm with a reinforcement called Q-learning:

1. Initialize $Q^{*}(s,a)$ arbitrarily. 2. Watch $s_{0}$ from the medium. 3. For $k=0,1,2,$... :

With the likelihood of $\varepsilon$ to choose the action $a_{k}$ accidentally, otherwise desirable: $a_{k}=\mathrm{argmax}_{a_{k}}Q^{*}(s_{k},a_{k})$ to send the action $a_{k}$ on Wednesday, to get a reward for the step $r_{k}$ and the following state of $s_{k+1}$. to update one cell of the table:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Adding a Neural Network

Finally, to move to algorithms capable of learning in complex MDPs with complex space states, you need to combine the classical theory of learning with reinforcement with the paradigms of deep learning.

Let’s admit we can’t afford to store $Q^{*}(s,a)$ as a memory table, for example if we play a video game and any images are submitted to us on the entrance. Then we can process any entrance signals available to the agent using a neural network $Q^{*}(s,a,\theta)$. For the same video games we easily process the screen image with a small spin network and issue for each possible action $a$ material scale $Q^{*}(s,a,\theta)$. Let’s also admit that the action space is still finite and small so that we can build such a strategy for a model, choose argxa ${{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{{}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}}

Let’s take a look at the Q-learning update formula for one shift $\left(s,a,r,s^{\prime}\right)$ :

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

The Q-learning theory suggested that the process of such a Q-function learning has much in common with the usual stochastic gradient drop. In this form, the formula suggests that, apparently,

$$ r+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})-Q_{k}^{*}(s,a) $$

This gradient compares the Belmanov target.

$$ r+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}) $$

With our current approximation of $Q_{k}^{*}(s,a)\mid$ and slightly adjusting this value by moving toward the target. Let’s try to “change” in this $\Phi$ ormule the Q function from the table representation on the neuron network.

Let’s consider this task of regression. To build one precedent for the learning sample, let’s take one we have a transition $\left(s,a,r,s^{\prime}\right)$. The entrance will be a pair of $s,a$. The target variable, the target, will be the Belmanov target

$$ y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta); $$

This is why Monte Carlo evaluates the right part of the Bellman optimity equation and is called a target. But it is important to remember that this target variable is actually “smalled”: in the formula is used taken from the transition $s^{\prime}$ which is only a sample of the transition function. In fact, we would like to study the average value of such a target variable, and therefore as a loss function we will take MSE. How will the step of the stainless gradient release look for the solution of this regression task (for just one presence)?

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

This practically accurately repeats the $\Phi$ Q-learning, which says that if the target is $r+$ $\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta)$ more than $Q^{*}(s,a,\theta)$, then we need to adjust the weight of our model so that $Q^{*}(s,a,\theta)$ becomes a little greater, and vice versa.

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

- to the right side of the Bellman optimity equation, that is, to model the method of simple iteration to solve the system of nonlinear equations.

The only difference between such a task of regression and those faced by traditional deep learning is that the target variable depends on our own model. Earlier the target variables were directly the source of the trainer signal. Now, when we want to study the future reward under the condition of optimal behavior, we don’t know this true value or even its stochastic assessments. Therefore we apply the idea of bootstrapping: we take the reward for the next step, and unfairly approach the rest of the reward by our current approximation $\operatorname*{max}_a{{{prime}} $Q^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If suddenly after a next step of optimization and updating the weights of the neuron network our model began to issue some slightly inadequate values, they risk to get into the target variable at the next step, we will take a step of training under inadequate target variables, the model will become even worse, and so on, the chain reaction will begin.

For stabilization, we use a trick called target network. Let’s make that we have a regression task changed not after each update of the neuron net, but at least once, let’s say, in 1000 steps of optimization. For this we introduce a complete copy of our neuron net (“target network”), the weight of which we will mark $\theta^{-}$. Every 1000 step we will copy the weight from our model to the target network $\theta^{-}\leftarrow\theta$, we will no longer change any more $\theta^{-$. When we want for the next transition $\left(s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s,s

$$ y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta^{-}) $$

Then the rule on which the target variable is built will change once in 1000 steps, and we 1000 steps will solve the same task of regression.

Experience Replay

To finally collect the Deep Q-learning algorithm (usually called DQN, Deep Qnetwork), we will need to take the last step linked again to the data collection. When we want to train a neuron network, we need for each weight update from somewhere to take a whole mini-batch of data, i.e. the batch of the transitions $\left(s,a,r,s^{\prime}\right)$ to average the gradient assessment by it. However, if we take the middle, we will make in it $N$ steps, then the $N$ transitions we encounter will be very similar to each other: they will all come from the same area of state space.

The first way available always when the environment is set with the help of a virtual simulator is the launch of parallel agents. It starts parallelly $N$ processes of agent interaction with the environment, and in order to collect another mini-batch transitions for learning, in all copies the one step of interaction is carried out, is gathered by one transition. Such a mini-batch will already be diverse.

More interesting second way. Let's go after a next step of interaction with the environment we won't immediately use the $\left(s,a,r,s^{\prime}\right)$ transition to update the model, but remember this transition and put it in the collection. Remember with all you encounter during the trials and errors transitions $\left(s,a,r,s^{\prime}\right)$ is called a replay buffer (replay buffer or experience replay). Now to update the weight of our network, we take and randomly sample from the equal distribution the desired amount of transitions from the whole history.

However, the use of replica buffer is possible far from all the learning algorithms with strengthening. The fact is that some learning algorithms with strengthening require that the data for the next step of weight updating are generated by the current, most fresh version of the strategy. Such algorithms refer to the class on-policy: they can improve the strategy only according to the data from it itself (“on policy”). An example of on-policy algorithms perform, for example, evolutionary algorithms. How they are structured: for example, you can set up populations strategies, play each with the environment, select the best and how to generate a new population (more detailed one of the most successful schemes in such a framework can be seen here).

And here’s the important point: Deep Q-learning, like the usual Q-learning, relates to off-policy learning algorithms with strengthening. It doesn’t matter which strategy, smart or not very, old or new, has generated the transition $\left(s,a,r,s^{\prime}\right)$, we still have to solve the Bellman optimity equation including for this pair of $s,a$ and we have enough when building the target only that $s^{\prime}$ is a sample of the transition function (and it’s just one depending on what strategy interacts in the environment). Therefore, to update the model of $Q^{{{s}(a$) we can fully produce the experience, meaning we can use the experience in that number.

(images/fdd4f4f98ab86534dbfa0e771a3f5c6da8a3708a2e04cf2bcff117b78b97f4a9.jpg) image source — UC Berkeley AI course

In any case, even in complex environments, when we interact with the environment we still have to somehow solve the exploration-exploitation dilemma, and use, for example, $\varepsilon$ -the desired research strategy. So, the DQN algorithm looks like:

Initialize the neural network $Q^{*}(s,a,\theta)$.

Initialize the target network by placing $\theta^{-}=\theta$.

Watch $s_{0}$ from the middle.

For $k = 0.1,2$... :

with the probability of $\varepsilon$ to choose the action $a_{k}$ by chance, otherwise desirable:

$$ a_{k}=\mathrm{argmax}_{a_{k}}Q^{*}(s_{k},a_{k},\theta) $$

send the action $a_{k}$ on Wednesday, get a reward for the step $r_{k}$ and the next state $s_{k+1}$. add the shift $(s_{k},a_{k},r_{k},s_{k+1})$ to the replica buffer. if the replica buffer has accumulated a sufficient number of shifts, take a training step. For this we sample a mini-batch of shifts $\left(s,a,r,s^{prime}\right)$ from the buffer. for each shift we consider the target variable: $y=r+gamma\operatorname*{max}{a_a_a_a_a_a_a_a_a_a_a_a_a_a_a_a_a

$$$$$$$$$$$$$$

If $k$ is divided into 1000, update the target network: $\theta^{-}\leftarrow\theta$.

The DQN algorithm does not require any handcrafted characters or specific settings under a given game. The same algorithm, with the same hyperparameters, can be launched on any of the 57 games of the ancient Atari console (example of the game in Breakout) and get any strategy. To compare the RL algorithms between them, the results are usually mediated across all the 57 games of the Atari. Recently, the algorithm named Agent57, combining quite a lot of modifications and improvements to the DQN and developing this idea, was able to defeat the person at once in all these 57 games.

What if the space of action is continuous?

Everywhere in the DQN we supposed that the space of action is discreet and small so that we can count the desirable strategy of $\pi(s)=\mathrm{argmax}_{a}Q^{*}(s,a,\theta) $ and count the maximum in the formula of the target variable of $\operatorname*{max}_{a}Q^{*}(s,a,\theta)$. If the space of action is continuously, and at each step from the agent is expected to choose several material numbers, then how to do this uncomprehensible. This situation occurs everywhere in the robotics. There every association of the robot can be, for example, rotated to the right / to the left, and such actions can be described by means of action in the dialect of the number of action in the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of the dialect of

And let’s, when we don’t know argmaxa $Q^{*}(s,a)$, get it closer to another neuron network. That is, get the second neuron network $\pi(s,\phi)$ with the parameters of $\phi$, and let’s teach it so that

$$$$$$$$$$$$$$$$$$$$$$$$

Well, we’ll take at each iteration of the algorithm the s status batch from our buffer replica and we’ll teach $\pi(s,\phi)$ to issue such actions on which our Q-function emits big scalar values:

$$ \sum_{s}Q^{*}(s,\pi(s,\phi),\theta){\to}\quad\operatorname*{max}_{\phi} $$

And since the actions are continuous, all left is differentiable and we can directly apply the most common backpropagation!

(images/5fec680b4e840851ae0268d87cba6c6ddeb4887b5264794c7eb7a8f562e71abc.jpg)

Now that there is an approximation of $\begin{array}{r l}{\pi(s,\phi)\approx}&{{{}\operatorname*{argmax}_{a}Q^(s,a,\theta)}\end{array}$, you can simply use it everywhere we need the maximum and maximum of our Q-function. We got the Actor-Critic scheme: we have an actor, $\pi(s,\phi)$ - a determined strategy, and the critic of $Q^{*}(s,a)$ who evaluates the choice of action by an actor and provides a gradient to improve it.

$$ y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta^{-})\approx r+\gamma Q^{*}(s^{\prime},\pi\left(s^{\prime},\phi\right),\theta^{-}) $$

This striking work euristic allows you to invent off-policy algorithms for continuous spaces of action; this approach includes algorithms such as DDPG, TD3 and SAC.

Policy Gradient algorithms

We learn from targets looking only one step forward using only s; this is a problem of accumulating error, because if between performing action and receiving $+1$ reward passes 100 steps, we need to "distribute" the received signal in a hundred steps.

$Q^{*}(s,a)$ instead of somewhat directly (“end-to-end”) remembering what actions in which conditions are good. Finally, our strategy is always determined when to interact with the environment during data collection, for example, we need a stochastic to guaranteed updating the Q-function for all s,a couples, and this problem had to be closed with castles.

There is a second approach to model-free RL algorithms, called Policy Gradient, which allows you to avoid the above shortcomings due to the on-policy mode of work. The idea looks like this: let's look for a strategy in the class of stochastic strategies, that is, let's introduce a neuron network modeling $\pi_{\boldsymbol{\theta}}(\boldsymbol{a}♰\mathrm{\Delta}s)$ directly. Then our functionality that we optimize,

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

We differentiate by the $\theta$ parameters, and the gradient is equal to:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

where $R_{t}$ - reward-to-go from the step $t$, that is, the prize collected in the play episode after the step $t$ :

$$$$$$$$$$$$$$$$$$$$

Sketch evidence

This formula tells us that the gradient of our function is also the mat. expectation on the trajectory. That is, we can try to count some assessment of this gradient by replacing the mat. expectation on the Monte Carlo assessment, and just start optimizing our function by the most common stochistic gradient drop! That is: we take our strategy $\pi_{\theta}$ with current values of the parameters θ, we play an episode (or a few) in the medium, that is, we sample $\tau\sim$ $\pi_{\theta}$ and then we take a step of gradient upward:

$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$$

Why this idea leads to an on-policy approach? For each step of a gradient step we must necessarily take $\tau\sim$ πθ with the freshest, with the current weights of $\theta$, and no other trajectory, born by any other strategy, will not fit us. Therefore, for each iteration of the algorithm we will have to again play another episode with the environment. This is sampleinefficient: inefficiently in the number of samples, we collect too much data and work very inefficiently with them.

Policy Gradient algorithms try to fight this inefficiency in a variety of ways, again by referring to the theory of assessment functions and boosted assessments that allow forecasting future awards without playing episodes completely to the end. Most of these algorithms remain on-policy mode and apply in any action spaces. These algorithms include such algorithms as Advantage Actor-Critic (A2C), Trust-Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO).

What else there?

We have yet understood model-free RL algorithms that were running without the knowledge of $p(s^{\prime})$ s,a ) and did not try to approach this distribution. However, in some spots the transition function is known to us: we know what state the environment will pass if we choose some action in such a state. It is clear that this information would be good to use somehow. There is an extensive model-based class that either suggests that the transition function is given, or we learn its approximation using $s,a$, $s^\prime}$ from our experience as a training choice. The AlphaZero algorithm based on this transition has been considered to be a game that is more difficult to play than this game, and we are able to play this game as a game of chess or a game of chess.

(images/2f04bdb68419d5ed92bd9370f2e7e5dbf0999e0a6cdd07688cc01ad5e750d32d.jpg)

Image source: UC Berkeley AI

Training with support aims to build algorithms capable of learning to solve any task presented in the MDP formalism. Like the usual optimization methods, they can be used in the form of a black box from finished libraries, for example, OpenAI Stable

Baselines. Inside such boxes, however, there will be quite a lot of hyperparameters that are not yet quite clear how to adjust to one or another practical task. And although the success of Deep RL shows that these algorithms are able to learn incredibly complex tasks such as victory over people in Dota 2 and StarCraft II, they require a huge amount of resources for this. Searching for more effective procedures is an open task in Deep RL.

In Shade there is a Practical RL course in which you will immerse yourself deeper into the world of deep learning with strengthening, learn more advanced algorithms and try to teach the neurons to solve different tasks in different environments.

The paragraph is not read

Note the paragraphs as read to see your learning progress

Join the handbook community Here you can find thinkers, experts and just interesting interlocutors. And more - get help or share knowledge.

to enter

$\bigcirc$ Notify an error

Previous paragraph

10.5 The task of ranking

The next paragraph

11.2 Crowdfunding

# AHneKc O6pa30BaHNe

Research Handbook Knowledge Base Event Journal Yandex Curriculum Yandex Person Yandex Practice School Data Analysis Programs at Universities About us Feedback Partners Information about educational organization User Agreement of handbooks

The Boot K