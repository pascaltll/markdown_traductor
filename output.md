# Content

Book of machine learning

![](images/d65809172d0d826cfd5df6d9ac9c02569ce45091ea65965c8e1edfed5dd273c4.jpg)

# 11.1 Training with support

The authors

![](images/32d40fbe70e1e4e4ab254a7a2aac15dfce151fb2bd65de2e54f43daff379842e.jpg)

# Ivanov Sergey

So far, the experience that made it possible to learn in our algorithms has been given in the form of a learning sample. How much such a learning model relates to the way one learns, for example, a person? To learn to ride a bicycle, cook a cake or play tennis, we don’t need huge dataset with examples of what to do at any time; instead we are able to learn by trial and error method, making attempts to solve a task, interacting with the surrounding world, and somehow improving our behavior based on the experience obtained during this interaction.

In learning with reinforcement (RL) we want to build an algorithm that modeles learning by the method of samples and errors. Instead of receiving a learning sample to the entrance such an algorithm will interact with some environment (environment), the surrounding world, and in the role of "marking" will be a reward (reward) - a scalary value that is issued after each step of interaction with the environment and shows how well the algorithm deals with the task set to it. For example, if you bak the cake, then for each cake you get$+1$And if you’re trying to ride a bicycle, then for every fall from the bicycle you’ll fly -1.

The prize doesn’t tell you exactly how to solve a task and what to do at all; the prize can be delayed in time (you’ve found a treasure in the desert, but to get the deserted cakes, you’ll still need a lot of time to get out of the desert; and the prize comes only for the cakes) or strongly cut (the majority of the time to give the agent)$+0$All this greatly distinguishes the task from learning with a teacher; the prize provides some “signal” for learning (good/bad), which is not, for example, in learning without a teacher.

![](images/d0f474ca25be1c278d43ecb12188a68e50c212944d13669312803aa6ccb1b0df.jpg)Image source: UC Berkeley AI

# Establishment of task

Now let’s try to formalize all this concept and get acquainted with the local terminology. The task of learning with support is set by the Markov Decision Process (MDP).$\left(\boldsymbol{\mathcal{S}},\boldsymbol{\mathcal{A}},\boldsymbol{\mathcal{P}},\boldsymbol{r} \right)$Where to:

S is the space of states (state space), a multitude of states in which any moment of time can be the environment. A is the space of action (action space), a multitude of options from which you need to make a choice at each step of your interaction with the environment.$\mathcal{P}$the transition function (transition function), which determines the change of the environment after the$s\in$ $s$The action was chosen.$a\in$ $\mathcal{A}$In general, the function of transitions can be stochastic, and then this function of transitions is modeled by distribution.$p(s^{\prime}\mid s,a)$What is the probability that the medium will move to what state after the choice of action?$a$in state$s$ .

$r{:}S\times$ $A\rightarrow\mathbb{R}$Reward function (reward function), which gives the scalary value for the choice of action$a$in the state s. This is our “learning signal”.

Traditionally, the subject interacting with the environment and influencing it is called in training with a support agent (agent). The agent is guided by some rule, also stochastic, how to choose actions depending on the current state of the environment, which is called a strategy (policy; the term is often transliterated and spoken policy) and is modeled by the distribution$\pi(a|\quad s)$Strategy and will be our search object, so as in classical machine learning, we are looking for some function.

Interaction with the agent’s environment with strategy$\pi(a|\quad s)$In the first place, the environment is in a state.$s_{0}$The agent simulates action from his strategy.$a_{0}\sim$ $\pi(a_{0}\mid s_{0})$The Wednesday responds to this by sampling its next state.$s_{1}\sim p\left(s_{1}\mid s_{0},a_{0}\right)\scriptscriptstyle{\boldsymbol{u}\boldsymbol{3}}$transfer functions, as well as awarding the agent a prize in the amount$r(s_{0},a_{0})$The process is repeated: the agent again simulates$a_{1}$The middle is the generation.$s_{2}$The Scale Prize$r(s_{1},a_{1})$This continues to infinity or until the medium passes into the terminal state, after entering which interaction is interrupted, and the collection by the agent of reward ends. If there are terminal states in the medium, one iteration of interaction from the initial state to entering the terminal state is called an episode (episode). The chain generated during the interaction of random sizes$s_{0}$ ​, $a_{0}$ ​, $s_{1}$ ​, $a_{1}$ ​, $s_{2}$ ​, $a_{2}$Note: the reward function can also be stochastic, and then the rewards by step will also be random sizes and part of the trajectory, but without limitation of the community we will consider the determined reward functions.

![](images/a70f06c857f6229545ccb1c99a6a8f9d583b743c9e041c9a236caaccdb2a2564.jpg)

So that,$\Phi$The active environment for us is the managed Markov chain: at every step we choose$a$We assume, first of all, the Markovic property: that the transition to the following state is determined only by the current state and does not depend on the entire previous history:

$$
p(s_{t+1}\mid s_{t},a_{t},s_{t-1},a_{t-1},\ldots,s_{0},a_{0})=p(s_{t+1}\mid s_{t},a_{t})
$$

Second, we assume stability: the function of transitions$p(s^{\prime}\mid s,a)$These are quite realistic assumptions: the laws of the world do not change over time (stacionary), and the state describes the world as a whole (marking). In this model of interaction there is only one unrealistic assumption: full observability (full observability), which states that the agent in his strategy$\pi(a|\quad s)$observes the whole state s completely and can choose action, knowing about the surrounding world absolutely everything; in reality we have only some partial observations of the state. This more realistic situation is modeled in partially observable MDP (POMDP), but we will further be limited to fully observed environments.

So we learned in the mathematical language to model the environment, the agent and their interaction with each other. It remains to understand what we want.$r_{t}=r(s_{t},a_{t})$However, states and actions$s_{t}$ ​, $a_{t}$The same agent can, by virtue of the stochastics both internal (by virtue of the random choice of action in its strategy), and external (by virtue of the stochastics in the function of transitions) to obtain a very different total reward$\textstyle\sum_{t\geq0}r_{t}$We will say that we want to learn how to choose actions so that we collect as much rewards as possible in average.

What does it mean in the average, in the average, in what? In the whole stochastic that is placed in our process of interaction with the environment.$\pi$the distribution of the trajectory in the space – with what probability the trajectory can meet us$\tau=$ $\left(s_{0},a_{0},s_{1},a_{1},\dots\right)$ :

$$
p({\mathcal T}|{\textit{\pi}})=p(s_{0},a_{0},s_{1},a_{1},\cdot\cdot\cdot|{\textit{\pi}})=\prod_{t\geq0}p(s_{t+1}|s_{t},a_{t}){\pi}(a_{t}|s_{t})
$$

This is according to this distribution we want to take the average of the agent receiving awards. They usually record it somehow like this:

$$
\mathbb{E}_{T\sim\pi}\sum_{t\geq0}r_{t}\to\operatorname*{max}_{\pi}
$$

Here mat. expectation on the trajectory is an infinite chain of mat. expectations:

$$
\mathbb{E}_{T\sim\pi}(\cdot)=\mathbb{E}_{a_{0}\sim\pi(a_{0}\mid s_{0})}\mathbb{E}_{s_{1}\sim p(s_{1}\mid s_{0},a_{0})}\mathbb{E}_{a_{1}\sim\pi(a_{1}\mid s_{1})}\dots(\cdot)
$$

This is the design we want to optimize by choosing a strategy.$\pi$In environments where interaction can last indefinitely long, the agent can learn to obtain an infinite reward, with which different paradoxes can be associated (for example, getting$+1$At each second step it becomes as good as getting$+1$Therefore, discounting (discounting) awards are introduced, which says: the cake is now better than the same

The reward we will receive in the future, the agent will discount for a certain number$\gamma$, smaller units. then our functional will take this kind:

$$
\mathbb{E}_{T\sim\pi}\sum_{t\geq0}\gamma^{t}r_{t}\to\operatorname*{max}_{\pi}
$$

![](images/ee89d39c3619039aeb8b6ee15ea83fc198582335fd617f049e7bd8a03a9ccda6.jpg)Image source: UC Berkeley AI

Let’s note that training with support is primarily the task of optimizing, optimizing functions of a particular type. If in classical machine learning the selection of the loss function can be considered an element of the engineering part of the solution, then here the reward function is assigned to us ready and determines the function that we want to optimize.

# Examples

The formalism of the MDP is very common, and it falls under almost everything that can be called a “intellectual task” (with the provision that it is not always clear which function of reward puts one or another task).

For example, the most simple examples of MDP can be drawn "on paper." For example, they are often considered "celled worlds" (GridWorlds): an agent is in some position of a fiber plate and can as an action choose one of the four directions. Such worlds can respond differently to the agent for the choice of action "to go to the wall", with a certain likelihood of moving the agent not in the direction he chose, containing objects in some cells and so on.

![](images/1e062fa68024bb497fa5eef0a680e2f63669bf82f497dae426959e29ecb57a37.jpg)Image source: UC Berkeley AI

A huge variety of MDP provides video games. You can think that the entrance agent will be submitted a picture of the video game screen, and several times a second the agent chooses which buttons on the controller he wants to press. Then the space of states - a lot of all kinds of images that can show you the video game. Multiple, in general, final (the final number of pixels of the screen with three color channels, each of which shows a total value from 0 to 255), but only very large; for example, they can no longer be listed or preserved all possible options in memory. But at each step you need to choose the action from the final to: which buttons to press, so this task is discreted management.

![](images/e590a4f7966ec2627b3c0037d613d81d0f1351b4d40707d6905679ab73a9da5a.jpg)

# Image source: UC Berkeley AI

Finally, the natural way to create the environment is the use of physical simulations. As a benchmark, locomotion is often used - the task of teaching any "existence" to walk within the framework of a physical model (examples can be seen, for example, here). Conceptually, within the framework of the task of learning with strengthening, we don't even matter how the simulation is organized or how the reward function is set: we want to build a common algorithm of optimization of this same reward. If the reward encourages the movement of the center of the mass "existence" along a certain direction, the agent will gradually learn to choose actions so that the existence is moving and not falling, if later it leads to the end of the episode and receives further reward.

![](images/5d0bf19a3fb6829475d535735516357593e60071be9780215b38ffad49dc588a.jpg)DeepMind Producing Flexible Behaviours in Simulated Environments

In such tasks, the agent at each step chooses several material numbers in the range [−1,1], where -1 - "maximally relax" the joint, and$+1$This space of action occurs in many robotics tasks, where you need to learn how to turn some roll, and it has the extreme right and the extreme left position, but you can choose any intermediate.

# OK, and how to decide?

It seems complicated, but mankind has already a lot of work on how to approach this as a very common task, with the basic idea you’ve probably already encountered.

The fact is that we optimize not to any functionality, but the average discounted cumulative reward. To invent a more effective solution than any approach that does not use this fact (e.g., evolutionary algorithms), we need to take advantage of the structure of the assigned task. This structure is set in the formalism of MDP and the definition of the process of interaction of the agent with the environment. Intuitively it is expressed as follows: here we sit in some state s and want to choose action$a$We know that after choosing this action we will receive a reward for this step.$r=r(s,a)$, the environment will put us into a state of s and, attention, further awaits us the submission of the equivalent structure: in precision the same task of choosing the optimal action, only in another state. Indeed: when we will make a decision on the next step, on the past we are no longer able to influence; stability means that the laws under which the environment is conducted have not changed, and markability says that history does not affect the future process of our interaction. This points to the thought that the task of maximizing the reward from the current state is closely linked to the task of maximizing the reward from the next state$s^{\prime}$Whatever it is.

To formulate this in the language of mathematics, the “additional variables” are introduced, the auxiliary values called assessment functions. Let’s get acquainted with one such assessment function – the optimal Q function that we’ll mark$Q^{*}(s,a)$Let’s say that$Q^{*}(s,a)$This is how much maximum rewards can be obtained (in average) after the choice of action.$a$From the state s. So:

$$
Q^{*}(s,a)=\operatorname*{max}_{\pi}\mathbb{E}_{\mathcal{T}\sim\pi\mid s_{0}=s,a_{0}=a}\sum_{t\geq0}\gamma^{t}r_{t}
$$

Recording${\mathcal{T}}\sim\pi|~s_{0}=s,a_{0}=a$This means that we are in a state.$s_{0}=s$Choose action$a_{0}=a$and then we continue to interact with the environment through a strategy$\pi$by generating thus the trajectory T. According to the definition, to count$Q^{*}(s,a)$You need to review all the strategies, see how many of them earn awards after choosing.$a$Therefore, this assessment function is called the optimal: it assumes that in the future after the choice of action$a$from state$s$The agent will behave optimally.

The definition is unconstructive, of course, because in reality we cannot do so, but it has an interesting property.$Q^{*}(s,a)$, then we know the optimal strategy. Really: imagine that you are able to$s$You have to make a choice from three actions, and you know the meanings.$Q^{*}(s,a)$You know that if you choose the first action$a=0$In the future, you will be able to get no more than we allow.$Q^{*}(s,a=0)=+3$You know that there is a strategy.$\pi$which is achieved to the maximum in determining the optimal Q function, that is, which actually allows you to obtain these$+3.$You know that if you choose the second action, then in the future you will be able to collect, allow,$Q^{*}(s,a=1)=+10$For the third action$Q^{*}(s,a=2)=-1.$The intuition suggests that you just have to choose action.$a=1$which will allow you to$+10,$There is no choice, but there is no choice, there is no choice, there is no choice.$a=1$This intuition does not deceive us, and the principle of such choice is called the principle of Bellman’s optimity.

The choice of the action at which the maximum is achieved by the actions of the Q function is called greedy (greedy) in relation to it. Thus, the principle of optimality of Bellman says:

The desired choice for the optimal Q function is optimal:

$$
\pi^{*}(s)=\operatorname{argmax}_{a}Q^{*}(s,a)
$$

Note: If the Q function reaches the maximum on several actions, you can choose any of them.

Let us note that this optimal strategy is determined. This interesting fact means that we generally do not need to look for a stochistic strategy. Our consideration for now even shows that we can simply try to find$Q^{*}(s,a)$, and then get out of it the optimal strategy, choosing action thirsty.

How to Search$Q^{*}(s,a)$Here on the scene and appears our observation of the structure of the task.$Q^{*}(s,a)$Really: Let’s consider a couple of state-action$s,a$On the one hand, according to the definition, we will be able in the future under the condition of optimal behavior to obtain$Q^{*}(s,a)$On the other hand, after we choose action$a$in state$s$We will receive a prize in one step.$\boldsymbol{r}(s,a)$The prize will be discounted on$\gamma$Wednesday will respond with a sample.$\mathbf{\boldsymbol{s}}^{\prime}\sim p(\mathbf{\boldsymbol{s}}^{\prime}\mid\mathbf{\boldsymbol{s}},\boldsymbol{a})$(the result of this sampling we can no longer influence and by this stochastic our future reward will have to be mediated), and then in a state of$s^{\prime}$In the presumption of the optimal behavior, we will choose that action.$a^{\prime}$where the maximum is achieved$Q^{*}(s^{\prime},a^{\prime})$In other words, after entering the$s^{\prime}$We can get$\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime})$The following recurrent ratio, called the Bellman Optimality Equation for Q-function, is true:

$$
Q^{*}(s,a)=r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime})
$$

We have a system of equations that bind values.$Q^{*}(s,a)$This is an unlinear system of equations, but it turns out that it is in some sense “good.” It has the only solution – and thus, the solution of this equation can be considered an equivalent definition.$Q^{*}(s,a)$The method of simple iteration solution of equation systems allows you to improve your current approach.$x$Resolution of a Type Equation$x=f(x)$to the right part. that is: initialize the arbitrary function$\begin{array}{r l}{Q_{0}^{*}(s,a)\colon S\times}&{{}A\to\mathbb{R},}\end{array}$which will be approaching$Q^{*}(s,a)$, then iterately we will put it in the right part of the Bellman optimity equations and upgrade our approach to the resulting values:

$$
\begin{array}{r l}{Q_{k+1}^{*}(s,a)\gets}&{{}r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})}\end{array}
$$

This process will lead us to the true$Q^{*}(s,a)$By the way, when you’ve met in the past with dynamic programming, you’ve most likely unclearly used this idea, except that often in the tasks to solve the equations of optimity of Bellman you can simply consistently exclude unknown variables; but the method of simple iteration gives a more general scheme, always applicable. And now for us it is the following: if we have some approach$Q^{*}$The calculation of the right part of the Bellman optimity equation will allow a better approach.

# Where is the method of trials and errors?

Resolve the method of simple iteration of the Bellman optimity equation and thus obtain$Q^{*}(s,a)$in reality it is possible only under two very significant restrictive conditions. It is necessary that, first of all, we can keep some current approximation$Q_{k}^{*}(s,a){\bf\delta B}$This is possible only if the spaces of states and actions are final and not very large, that is, for example, in your MDP only 10 states and 5 actions, then$Q^{*}(s,a)-\mathsf{s T o}$But what if you want to learn to play a video game, and the state is an input image? Then a lot of images that can show you the video game, save in memory will no longer succeed. Well, let’s admit that the number of states and the number of actions is not very large, and we still can save the table in memory, and later we’ll take this limit by modeling$Q^{*}(s,a)$With the help of the neuron.

Secondly, we need to be able to count the expression that stands to the right in the Bellman optimity equation:

$$
r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})
$$

Few things that in difficult environments take mat. expectation on the function of transitions$\mathbb{E}_{s^{\prime}\sim p\left(s^{\prime}\mid s,a\right)}$Imagine you ride a bicycle: can you, according to the current state of the surrounding world, for example, the position of all the atoms in the universe, tell us with what chances the world will be in the next moment of time? This consideration also suggests that it would be good if we could solve a task, avoiding even the attempt to study this complex function of transitions.

What is available to us? we can take some strategy$\pi$(important point: we have to choose which) and interact it with the environment. “Try to solve a task.” We can generate with the help of$\pi$a whole trajectory or even do just one step in the environment. Thus we will collect data: allow, we were in a state of s and made the choice of action$a$We will know what the reward is.$r=r(s,a)$We get for this step and, most importantly, in what state.$s^{\prime}$We were transferred to the middle.$s^{\prime}$Sample from the function of transition$\mathbf{\boldsymbol{s}}^{\prime}\sim p(\mathbf{\boldsymbol{s}}^{\prime}\mid\mathbf{\boldsymbol{s}},\boldsymbol{a})$The information collected is four.$\left(s,a,r,s^{\prime}\right)-$It is called a transition (transition) and can somehow be used to optimize our strategy.$\left(s,a,r,s^{\prime}\right)$With only samples in the hands.$s^{\prime}\sim$ $p(s^{\prime}\mid s,a)$How to use a dynamic programming scheme? What if we are going to change the value$Q_{k}^{*}(s,a)$Not on

$$
r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}),
$$

which we cannot count, and on his Monte Carlo assessment:

$$
r(s,a)+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}),
$$

Where$s^{\prime}$- a sample of the transition function from the experience we have collected? in average, such a replacement is true. such a Monte Carlo assessment of the right part for the given transition$\left(s,a,r,s^{\prime}\right)$It is called the Belmanov target, i.e. the "target variable."Why this name - we will see a little later.

To understand how we need to act, we will consider some typical situation.$a$From a certain state$s$Wednesday is rewarding$r(s,a)=0$and puts us in equal possibility.$s^{\prime}$for whom$\begin{array}{r}{\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})=+1,}\end{array}$in the state.$s^{\prime}$for whom$\begin{array}{r}{\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})=-1.}\end{array}$The simple iteration method says that the next iteration needs to be replaced.$Q_{k}^{*}(s,a)\mathsf{H a}0.5\gamma$ ⋅ $(+1)+$ $0.5\gamma$ ⋅ $(-1)=0$But in reality, we will only find one outcome, and the target – Monte Carlo’s assessment of the right part of the Bellman’s optimity equation – will be 0.5 equal.$+\gamma$with a probability of 0.5 equal$-\gamma$It is clear that we cannot simply take and hardly replace our current approach.$Q_{k}^{*}(s,a)$on the counted Belmanovsky targets on some one transition, as we could have been lucky (we saw$+\gamma$(or not to be lucky)$-\gamma)$Let us instead do the same as the average is taught on the sample: not to move our current approximation "hardly" to the value of the next sample, but to mix the current approximation with the next sample.$\left(s,a,r,s^{\prime}\right)$We do not replace$Q_{k}^{*}(s,a)$to the stochistic assessment of the right part of the Bellman equation of optimity, and we just move toward it:

$$
Q_{k+1}^{*}(s,a)\gets~(1-\alpha)Q_{k}^{*}(s,a)+\alpha(r+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}))
$$

Thus, we carry out the exponential smoothing of the old approach.$Q_{k}^{*}(s,a)$and a new assessment of the right part of the Bellman optimity equation with a fresh sample$s^{\prime}$The choice$\alpha$Here it determines how much we pay attention to the last samples, and has the same physical meaning as the learning rate.$s^{\prime}$We will move to the side.

$$
r(s,a)+\gamma\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}),
$$

And that means to apply the "smalled" method of simple iteration.

So, the next idea arises. We will somehow interact with the environment and collect passages$\left(s,a,r,s^{\prime}\right)$For each transition, we will update one cell in our Q-table size number of states to the number of actions according to the above formula. Thus we will get as a "smalled" method of simple iteration, where we only update one cell in each step of the table, and do not hardly replace the value to the right part of the equations.

optimality, but we just move in some in the average correct stochastic direction.

It is very similar to a stochistic optimization like a stochistic gradient drop, and therefore the matching guarantees look similar.$Q^{*}(s,a)$For any couple.$s,a$We carry out an infinite number of updates throughout the process, and the learning rate (hyperparameter)$\alpha$) in them behaves as learning rate from the conditions of consistency of the stochastic gradient drop:

$$
\sum_{i}\alpha_{i}=+\infty,\qquad\sum_{i}\alpha_{i}^{2}<+\infty
$$

# The example

Kolobok Kolabulka likes to play in the lottery. Let’s admit, in some state s he performed the action$a$Buying a ticket and suggesting that you will be able to get it in the future$Q^{*}(s,a)=+100$However, for the purchase of the ticket he pays $10 and thus loses $10 awards on this step.$r(s,a)=-10$At the same time, getting into a state$s^{\prime}=s$where he is again offered to buy a ticket to the lottery (he can choose the action "buy" or the action "not buy").$0<+100$This is why the column suggests that in the future$s^{\prime}$He can get M.$\arg Q^{*}(s^{\prime},a^{\prime})=+100.$for simplicity.$\gamma=+1$Then it turns out that the one-step approach to the future prize$r(s,a)+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime})=-10+100=90.$Yes →$s^{\prime}$here - a random size, the column could be lucky or not lucky (and we, having seen only one sample of the function of transitions, can't say with certainty whether we were lucky now or not), but our$\Phi$Ormula says to move the approximation$Q^{*}(s,a)$to the Belmanov Target.

$\begin{array}{c c c}{{s\longrightarrow}}&{{\displaystyle{\begin{array}{c}{\circ^{\circ}{\frac{\circ^{\circ}\left(Q^{\star}(s,a)=+100\right)}{a::C^{\circ}8\mathrm{BL}}}}\ {{-}}\end{array}}}}&{{r(s,a)=-10}}\ {{\longrightarrow}}&{{Q^{\star}(s^{\prime},a^{\prime}=40!)=+100}}\ {{Q^{\star}(s^{\prime},a^{\prime}==5\mathrm{B}!)=0}}\end{array}$ $\mathring{\mathfrak{z}^{\star}}(s,a)\longleftarrow(1-\alpha)Q^{\star}(s,a)+\alpha\big(r(s,a)+\mathop{m a x}_{a^{\prime}}Q^{\star}(s^{\prime},a^{\prime})\big)=(1-\alpha)\bullet100$

The Learning Rate$\alpha=0.5$: then, moving +100 toward$+90$The expectations of the future prize after the purchase of a lottery ticket fall to 95.$+95>0$So the bull seems that buying a ticket is more profitable than not buying, so let's consider the next passage. Let's admit that the bull bought the ticket again, lost $10 again and got the same again$s^{\prime}=s.$Our update will say to reduce value again.$Q^{*}(s,a)$ :

![](images/46ad0328877815b26d19342967c906411a503ccb5937922bbeeb87b0a2057efc.jpg)

It is clear that if the column continues so hard, the target will be 10 times less than the current approximation, and$Q^{*}(s,a)$It will all decrease and decrease until it falls down to zero (and there will be more profitable not to buy a ticket anymore).$s^{\prime}$The matching victory in the lottery (and this is apparently happening with some small probability), the target will become very big, and the approximation$Q^{*}(s,a)$Our update will say to increase significantly:

![](images/d54fdd3121ba7c58b8c1d2b9c85dbcb3a1c63370080eb125e0e17592592357b8.jpg)

Let’s assume that the medium for buying a lottery ticket meets the probability.$p$Return to the same state.$s^{\prime}=s$where the column is offered to buy another ticket, and with the probability of 1–$p$The ticket is profitable, and the bull is in such a state.$s^{\prime}$In which he can take the prize and get it.$+1000$(After this interaction with the environment, let’s say, ends.) Let’s record the Bellman optimity equation for action$a$Buy a ticket in a state of s:

$$
Q^{*}(s,a)=r(s,a)+\gamma\left(p\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime}=s,a^{\prime})+(1-p){\cdot}(+1000\mathrm{~})\right)
$$

here$\operatorname*{max}_{a^{\prime}}$ ​ $Q^{*}(s^{\prime}=s,a^{\prime})=\operatorname*{max}(Q^{*}(s,a),0)$, since the bulldozer can either buy a ticket or not buy (this, let's admit, will bring him 0 rewards). It's clear that if the purchase of a ticket does not bring more than 0 rewards, then there's no sense to buy it. By putting all the numbers from the example, we get:

$$
Q^{*}(s,a)=-10+p\mathrm{max}\left(Q^{*}(s,a),0\right)+1000(1-p)
$$

The chance of losing in the lottery.$p=0.99$The solution of the equation is$Q^{*}(s,a)=0.$In this case, the action "buy a ticket" and "not buy" are equal, and both in the future will bring an average of 0 awards.$p>0.99$Buying a ticket is not profitable.$p<0.99$Although the target contains its own current approximation of the future prize and only one sample s is used instead of a fair average on all kinds of outcomes,$\Phi$The ormule of updating will gradually come to this solution. With the column the real meaning$p$Unknown, and in the update formula this probability only affected the appearance of one or another.$s^{\prime}$in another target.

This algorithm, to which we have already practically come, is called Q-learning, “learning of optimal Q-function.” However, we still have to answer one question: how do we need to collect data to meet the requirements for compatibility? How do we interact with the environment so that we each cell s,and not cease to update?

# The Exploration-Exploitation

The task of multi-hand bandits that met there is actually a private case of the training task with strength, in which after the first choice of action the episode is guaranteed to be completed, and this private case of the task is often used to study this dilemma. Let's consider this dilemma in our context.

Per, on the next step of the algorithm we have some approach.$Q_{k}(s,a)\approx$ $Q^{*}(s,a)$Approaching this, of course, is inaccurate, as the algorithm, if it falls to the real optimal Q function, then on infinity. How do you need to interact with the environment? If you want to get the maximum reward, you probably deserve to take advantage of our theory and engage in exploitation by choosing an action thirsty:

$$
\pi(s)=\operatorname{argmax}_{a}Q_{k}(s,a)
$$

Unfortunately, such a choice is not a fact that coincides with the real optimal strategy, and the most important thing, it is determined. This means that when this strategy interacts with the environment, many couples will never meet simply because we never choose action$a$In a state of s. And then we, it turns out, risk not to update the cell again.$Q_{k}(s,a)$For such couples.

Such situations can easily lead to the trigger of the algorithm. We wanted to learn to ride a bicycle and got$+0.1$After the first trials and errors we found that cycling brings us -5, as we very soon crashed into the trees and updated our approximation of the Q-function samples with a negative reward; but if we don’t even take a bicycle and simply take nothing to do, then we’ll be able to avoid the trees and we’ll get 0. Just because in our strategy of interaction with the environment they never met$s,a$That leads to a positive reward, and a thirsty strategy towards our current Q-function approximation never chooses them. Therefore we need to experiment and try new options.

The exploration mode implies that we interact with the environment through any stochistic strategy.$\forall s,a\colon\pi(a|\quad s)>0.$For example, such a strategy is a random strategy choosing random actions. How strange, the collection of experience using random strategy allows you to travel with a non-null probability in all areas of state space, and theoretically even our Q-function learning algorithm will match. Does this mean that the exploration is enough, and the exploitation can be hit?

In reality, we understand that to get to the most interesting areas of state space where the reward function is the largest, not so simple, and the random strategy will do so with a non-null probability, but the probability will be exponentially small.$Q_{k}(s,a)$For these interesting states infinite many times, that is, we will have to wait for an unusual rare venue far from a few times. Where it is more reasonable to use the already available knowledge and with the help of a thirsty strategy that already knows something to go to these interesting states. Therefore, to solve the dilemma exploration-exploitation usually take our current thirsty strategy and something with it do so that it becomes a little random. For example, with the probability$\varepsilon>0$Choose a random action, and with the probability$1-$Then we use the knowledge more often, and we choose any action with a non-zero probability; such a strategy is called$\varepsilon$It is desirable, and it is the easiest way to solve this dilemma.

Let’s assert what we have achieved, in the form of a table learning algorithm with a reinforcement called Q-learning:

1 Initialization$Q^{*}(s,a)$Voluntary 2. to observe$s_{0}$On Wednesday, 3 for$k=0,1,2,$ … :

with probability$\varepsilon$Choose action$a_{k}$Unfortunately, otherwise it is thirsty:$a_{k}=\mathrm{argmax}_{a_{k}}Q^{*}(s_{k},a_{k})$sending action$a_{k}$On Wednesday, we will receive a prize for a step.$r_{k}$The next state$s_{k+1}$Update one cell of the table:

$$
Q^{*}(s_{k},a_{k})\gets(1-\alpha)Q^{*}(s_{k},a_{k})+\alpha(r_{k}+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s_{k+1},a^{\prime}))
$$

# Adding a Neuroscience

Finally, to move to algorithms capable of learning in complex MDPs with complex space states, you need to combine the classical theory of learning with reinforcement with the paradigms of deep learning.

Let’s admit, we can’t let ourselves keep.$Q^{*}(s,a)$as a memory table, for example, if we play a video game and any images are submitted to us on the entrance. Then we can process any entrance signals available to the agent using a neuron network$Q^{*}(s,a,\theta)$For the same video games, we easily process the screen image with a small scratch and issue for each possible action.$a$The material scale$Q^{*}(s,a,\theta)$Let’s also admit that the space of action is still final and small so that we can build a desirable strategy for such a model, choose argmaxa$Q^{*}(s,a,\theta)$How to teach a neuron network?

Let’s look at the Q-learning upgrade formula for one transition.$\left(s,a,r,s^{\prime}\right)$ :

$$
\begin{array}{r l r}&{}&{Q_{k+1}^{*}(s,a)\gets(1-\alpha\mathrm{\boldmath~\Psi~})Q_{k}^{*}(s,a)+\alpha(r+\gamma\mathrm{\boldmath~\Psi~}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime}))=}\ &{}&{=Q_{k}^{*}(s,a)+\alpha(r+\gamma\mathrm{\boldmath~\Psi~}\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})-Q_{k}^{*}(s,a))}\end{array}
$$

The Q-learning theory suggested that the process of such a Q-function learning has much in common with the usual stochastic gradient drop. In this form, the formula suggests that, apparently,

$$
r+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})-Q_{k}^{*}(s,a)
$$

This gradient compares the Belmanov target.

$$
r+\gamma\operatorname*{max}_{a^{\prime}}Q_{k}^{*}(s^{\prime},a^{\prime})
$$

With our current approach$Q_{k}^{*}(s,a)\mid$and slightly adjust this value by moving toward the target. Let's try to "change" in this$\Phi$Q-function with a table representation on the neuron network.

Let’s consider this task of regression. To build one precedent for the learning sample, let’s take one that we have.$\left(s,a,r,s^{\prime}\right)$The entrance will be a couple.$s,a$The target variable, the target, will be the Belmanov target

$$
y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta);
$$

Dependency from Parameters$\theta$This is why Monte Carlo assesses the right part of the Bellman optimity equation and is called a target. But it is important to remember that this target variable is actually "smalled": the formula uses the transition taken from the transition$s^{\prime}$In fact, we would like to study the average value of such a target variable, and therefore as a loss function we will take MSE. How will the step of the stochastic gradient descent look to solve this task of regression (for simplicity - for one precedent)?

$$
\begin{array}{r l}&{\partial_{k+1}\leftarrow\theta_{k}-\alpha\nabla_{\theta}(y-\mathrm{\bf~Q}^{*}(s,a,\theta))^{2}=}\ &{\qquad=\theta_{k}+2\alpha(y-\mathrm{\bf~}Q^{*}(s,a,\theta))\nabla_{\theta}Q^{*}(s,a,\theta)=}\ &{\qquad=\theta_{k}+2\alpha(r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta)-\mathrm{\bf~}Q^{*}(s,a,\theta))\nabla_{\theta}Q^{*}(s,a,\theta)}\end{array}
$$

This is almost accurately repeated.$\Phi$Q-learning, which says that if you are targeted$r+$ $\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta)$more$Q^{*}(s,a,\theta)$We need to set the weight of our model so that$Q^{*}(s,a,\theta)$It has become a bit greater, and vice versa. in the average with such optimization we will move toward

$$
\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}y=\mathbb{E}_{s^{\prime}\sim p(s^{\prime}|s,a)}\left[r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta)\right]
$$

- to the right side of the Bellman optimity equation, that is, to model the method of simple iteration to solve the system of nonlinear equations.

The only difference between this task of regression and those faced by traditional deep learning is that the target variable depends on our own model. Earlier the target variables were directly the source of the learning signal. Now, when we want to study the future reward under the condition of optimal behavior, we do not know this true value or even its stochastic assessments. Therefore we apply the idea of bootstrapping: we take the reward for the next step, and unfairly approach the rest of the reward by our current approximation$\operatorname*{max}_{a^{\prime}}$ $Q^{*}(s^{\prime},a^{\prime},\theta)$Yes, behind this is the idea of the method of simple iteration, but it is important to understand that such a target variable only indicates the direction for learning, but it is not a true approximation of future awards or even their unlocated assessment. Therefore, they say that in this task of regression very shifted (biased) target variables.

If suddenly after a next step of optimization and updating the weights of the neuron network our model began to issue some slightly inadequate values, they risk to get into the target variable at the next step, we will take a step of training under inadequate target variables, the model will become even worse, and so on, the chain reaction will begin.

To stabilize, we use a trick called the target network (target network). Let’s make it that we have the task of regression changed not after each update of the neuron networks weights, but at least once, let’s say, in 1000 steps of optimization. For this we’ll make a complete copy of our neuron networks (target network), the weight of which we’ll mark$\theta^{-}$Every 1000 steps we will copy the weight from our model to the target network.$\theta^{-}\leftarrow\theta$No more to change$\theta^{-}$When we want for another transition$\left(s,a,r,s^{\prime}\right)$to build a target, we will not use our fresh model, but a target network:

$$
y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta^{-})
$$

Then the rule on which the target variable is built will change once in 1000 steps, and we 1000 steps will solve the same task of regression.

# Experience Replay  

To finally collect the Deep Q-learning algorithm (usually called DQN, Deep Qnetwork), we will need to take the last step linked again to data collection. When we want to train the neuron network, we need for each weight update somewhere to take a whole mini-batch of data, i.e. the batch of transitions$\left(s,a,r,s^{\prime}\right)$to mediate the gradient assessment by it. However, if we take the medium, we will do it$N$The steps we met$N$The transitions will be very similar to each other: they will all come from the same area of state space. The training of a neuron on correlated data is a bad idea, as such a model will quickly forget what it has taught on past iterations.

The first way available always when the environment is set using a virtual simulator is the launch of parallel agents.$N$The processes of interaction of the agent with the environment, and in order to collect the next mini-batch transitions for learning, in all copies is carried out by one step of interaction, is gathered by one transition.

After another step of interaction with the environment we will not immediately use the transition$\left(s,a,r,s^{\prime}\right)$to update the model, and remember this transition and put it in the collection. Remember with all you met during the trials and errors transitions$\left(s,a,r,s^{\prime}\right)$Now, in order to update the weight of our network, we take and randomly sampling the desired number of transitions from the entire history.

However, the use of replica buffer is possible far from all the learning algorithms with strengthening. The fact is that some learning algorithms with strengthening require that the data for the next step of weight updating are generated by the current, most fresh version of the strategy. Such algorithms refer to the class on-policy: they can improve the strategy only according to the data from it itself (“on policy”). An example of on-policy algorithms perform, for example, evolutionary algorithms. How they are structured: for example, you can set up populations strategies, play each with the environment, select the best and how to generate a new population (more detailed one of the most successful schemes in such a framework can be seen here).

And here’s the important point: Deep Q-learning, like the usual Q-learning, relates to off-policy learning algorithms with strengthening. It doesn’t matter which strategy, smart or not very, old or new, has generated a transition.$\left(s,a,r,s^{\prime}\right)$We still need to resolve the Bellman equation of optimality, including for this pair.$s,a$It is enough for us to build a target.$s^{\prime}$has been a sample of the transition function (and it is just one regardless of what strategy interacts in the environment).$Q^{*}(s,a)$We can by a completely arbitrary experience, and therefore we can also use experience replay.

![](images/fdd4f4f98ab86534dbfa0e771a3f5c6da8a3708a2e04cf2bcff117b78b97f4a9.jpg)Image source: UC Berkeley AI

In any case, even in complex environments, when we interact with the environment we still have to somehow solve the exploration-exploitation dilemma, and use, for example,$\varepsilon$So, the DQN algorithm looks like this:

Initialization of the neuron network$Q^{*}(s,a,\theta)$ .

Initializing the target network, putting$\theta^{-}=\theta$ .

3 to observe$s_{0}$from the middle.

4 for$k=0,1,2,$ … :

with probability$\varepsilon$Choose action$a_{k}$Unfortunately, otherwise it is thirsty:

$$
a_{k}=\mathrm{argmax}_{a_{k}}Q^{*}(s_{k},a_{k},\theta)
$$

sending action$a_{k}$On Wednesday, we will receive a prize for a step.$r_{k}$The next state$s_{k+1}$Add a transition$(s_{k},a_{k},r_{k},s_{k+1})$if the replica buffer has accumulated a sufficient number of passes, take a step of training. For this we sample a mini-batch of passes$\left(s,a,r,s^{\prime}\right)$For each transition we consider the target variable:$y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta^{-})$to take a step of gradient down to upgrade θ, minimizing

$$
{\sum}(y-~Q^{*}(s,a,\theta~))^{2}
$$

If$k$Divided by 1000, update the target network:$\theta^{-}\leftarrow\theta$ .

The DQN algorithm does not require any handcrafted characters or specific settings under a given game. The same algorithm, with the same hyperparameters, can be launched on any of the 57 games of the ancient Atari console (example of the game in Breakout) and get any strategy. To compare the RL algorithms between them, the results are usually mediated across all the 57 games of the Atari. Recently, the algorithm named Agent57, combining quite a lot of modifications and improvements to the DQN and developing this idea, was able to defeat the person at once in all these 57 games.

# What if the space of action continues?

Everywhere in the DQN we supposed that the action space is discreet and small so we can consider a thirsty strategy$\pi(s)=\mathrm{argmax}_{a}Q^{*}(s,a,\theta) $and calculate the maximum in the target variable formula$\operatorname*{max}_{a}Q^{*}(s,a,\theta)$If the space of action is continuous, and on each step from the agent is expected to choose several material numbers, then how to do this is uncomprehensible. Such a situation occurs everywhere in the robotics. There every robot association can, for example, be turned to the right/left, and such actions are easier to describe by a set of numbers in the range [-1, 1], where -1 is the extreme left position,$+1$- Extremely right, and any intermediate options are available. At the same time, the discretion of actions is not an option due to the exponential explosion of the number of options and the loss of the semanticity of actions. We, in general, need in DQN only one problem to solve: somehow to learn the maximum of actions to take

And let's go when we don't know argmaxa$Q^{*}(s,a)$We’ll get it closer to another neuron network, that is, we’ll get the second neuron network.$\pi(s,\phi)$with parameters$\phi$And we will teach her so that

$$
\begin{array}{r}{\pi(s,\phi)\approx\mathrm{~argmax}_{a}Q^{*}(s,a,\theta).}\end{array}
$$

Well, we’ll be on every iteration of the algorithm to take a batch state s from our buffer replica and we’ll learn$\pi(s,\phi)$to issue such actions on which our Q function emits large scalary values:

$$
\sum_{s}Q^{*}(s,\pi(s,\phi),\theta){\to}\quad\operatorname*{max}_{\phi}
$$

And since the actions are continuous, all left is differentiable and we can directly apply the most common backpropagation!

![](images/5fec680b4e840851ae0268d87cba6c6ddeb4887b5264794c7eb7a8f562e71abc.jpg)

Now that there is an approximation$\begin{array}{r l}{\pi(s,\phi)\approx}&{{}\operatorname*{argmax}_{a}Q^{*}(s,a,\theta)}\end{array}$We can simply use it anywhere where we need the maximum and maximum of our Q function. We got the Actor-Critic scheme: we have an actor,$\pi(s,\phi)$Defined Strategy and Critic$Q^{*}(s,a)$The actor learns to choose the actions that most like criticism, and the critic learns by regression with the target variable.

$$
y=r+\gamma\operatorname*{max}_{a^{\prime}}Q^{*}(s^{\prime},a^{\prime},\theta^{-})\approx r+\gamma Q^{*}(s^{\prime},\pi\left(s^{\prime},\phi\right),\theta^{-})
$$

This striking work euristic allows you to invent off-policy algorithms for continuous spaces of action; this approach includes algorithms such as DDPG, TD3 and SAC.

# Policy Gradient algorithms

We learn from targets looking only one step forward using only s ; it is a problem of accumulating error, because if between performing action and receiving a reward$+1$100 steps passes, we need a hundred steps to “distribute” the received signal.

$Q^{*}(s,a)$Finally, our strategy is always determined when to interact with the environment during data collection, for example, we need a stochastic to guaranteed updating the Q-function for all s,a couples, and this problem had to be closed with castles.

There is a second approach to model-free RL algorithms, called Policy Gradient, which allows you to avoid the above disadvantages due to the on-policy mode of work. The idea looks like this: let's look for a strategy in the class of stochastic strategies, that is, let's introduce a neuron network, modeling$\pi_{\boldsymbol{\theta}}(\boldsymbol{a}|\mathrm{\Delta}s)$The functionality that we optimize,

$$
J(\theta)=\mathbb{E}_{T\sim\pi_{\theta}}\sum_{t\geq0}\gamma^{t}r_{t}\to\operatorname*{max}_{\theta},
$$

Differentiate by Parameters$\theta$The gradient is equal:

$$
\nabla_{\theta}J(\theta)=\mathbb{E}_{T\sim\pi_{\theta}}\sum_{t\geq0}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})\gamma^{t}R_{t},
$$

Where$R_{t}$Reward-to-go by step$t$This is the prize collected in the play episode after the step.$t$ :

$$
R_{t}=\sum_{t^{\prime}\geq t}\gamma^{t^{\prime}-t}r_{t^{\prime}}
$$

# Sketch evidence

This formula tells us that the gradient of our function is also the mat. expectation on the trajectory. And thus, we can try to count some assessment of this gradient by replacing the mat. expectation on the Monte Carlo assessment, and just start optimizing our function by the most usual stochastic gradient drop! And that is: we take our strategy$\pi_{\theta}$with the current values of the parameters θ, we play an episode (or several) in the environment, i.e. sampling$\tau\sim$ $\pi_{\theta}$And then we take a step of gradient lifting:

$$
\theta\gets\theta+\alpha\sum_{t\geq0}\nabla_{\theta}\log\pi_{\theta}(a_{t}\mid s_{t})\gamma^{t}R_{t}
$$

Why this idea leads to an on-policy approach? For every step of a gradient step we must necessarily take$\tau\sim$πθ with the freshest, with current weights$\theta$Therefore, for each iteration of the algorithm we will have to play another episode with the environment again. This is sampleinefficient: inefficient in the number of samples, we collect too much data and work very inefficiently with them.

Policy Gradient algorithms try to fight this inefficiency in a variety of ways, again by referring to the theory of assessment functions and boosted assessments that allow forecasting future awards without playing episodes completely to the end. Most of these algorithms remain on-policy mode and apply in any action spaces. These algorithms include such algorithms as Advantage Actor-Critic (A2C), Trust-Region Policy Optimization (TRPO) and Proximal Policy Optimization (PPO).

# What else there?

We have so far understood the model-free RL algorithms that were running without knowledge of$p(s^{\prime})$s,a ) and have not tried to approach this distribution. However, in some spots the function of the transition is known to us: we know what state the environment will pass if we choose some action in such a state. It is clear that this information would be good to use somehow. There is an extensive class of model-based that either suggests that the function of the transition is given, or we learn its approach using$s,a$ , $s^{\prime}$The AlphaZero algorithm on the basis of this approach exceeded the person in the Go game, which was considered a much more complicated game than chess; and this algorithm is possible to start training on any game: both on Go and on chess or chess.

![](images/2f04bdb68419d5ed92bd9370f2e7e5dbf0999e0a6cdd07688cc01ad5e750d32d.jpg)

# Image source: UC Berkeley AI

Training with support aims to build algorithms capable of learning to solve any task presented in the MDP formalism. Like the usual optimization methods, they can be used in the form of a black box from finished libraries, for example, OpenAI Stable

Baselines. Inside such boxes, however, there will be quite a lot of hyperparameters that are not yet quite clear how to adjust to one or another practical task. And although the success of Deep RL shows that these algorithms are able to learn incredibly complex tasks such as victory over people in Dota 2 and StarCraft II, they require a huge amount of resources for this. Searching for more effective procedures is an open task in Deep RL.

In Shade there is a Practical RL course in which you will immerse yourself deeper into the world of deep learning with strengthening, learn more advanced algorithms and try to teach the neurons to solve different tasks in different environments.

# The paragraph is not read

Note the paragraphs as read to see your learning progress

Join the handbook community Here you can find thinkers, experts and just interesting interlocutors. And more - get help or share knowledge.

to enter

$\bigcirc$Reporting an error

Previous paragraph

10.5 The task of ranking

The next paragraph

11.2 Crowdfunding

# AHneKc O6pa30BaHNe  

Research Handbook Knowledge Base Event Journal Yandex Curriculum Yandex Person Yandex Practice School Data Analysis Programs at Universities About us Feedback Partners Information about educational organization User Agreement of handbooks

The Boot K