# -*- coding: utf-8 -*-
"""Learning rules.

This module contains classes implementing gradient based learning rules.
"""

import numpy as np


class GradientDescentLearningRule(object):
    """Simple (stochastic) gradient descent learning rule.

    For a scalar error function `E(p[0], p_[1] ... )` of some set of
    potentially multidimensional parameters this attempts to find a local
    minimum of the loss function by applying updates to each parameter of the
    form

        p[i] := p[i] - learning_rate * dE/dp[i]

    With `learning_rate` a positive scaling parameter.

    The error function used in successive applications of these updates may be
    a stochastic estimator of the true error function (e.g. when the error with
    respect to only a subset of data-points is calculated) in which case this
    will correspond to a stochastic gradient descent learning rule.
    """

    def __init__(self, learning_rate=1e-3):
        """Creates a new learning rule object.

        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.

        """
        assert learning_rate > 0., 'learning_rate should be positive.'
        self.learning_rate = learning_rate

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        This must be called before `update_params` is first called.

        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
        self.params = params

    def reset(self):
        """Resets any additional state variables to their intial values.

        For this learning rule there are no additional state variables so we
        do nothing here.
        """
        pass

    def update_params(self, grads_wrt_params):
        """Applies a single gradient descent update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        for param, grad in zip(self.params, grads_wrt_params):
            param -= self.learning_rate * grad


class MomentumLearningRule(GradientDescentLearningRule):
    """Gradient descent with momentum learning rule.

    This extends the basic gradient learning rule by introducing extra
    momentum state variables for each parameter. These can help the learning
    dynamic help overcome shallow local minima and speed convergence when
    making multiple successive steps in a similar direction in parameter space.

    For parameter p[i] and corresponding momentum m[i] the updates for a
    scalar loss function `L` are of the form

        m[i] := mom_coeff * m[i] - learning_rate * dL/dp[i]
        p[i] := p[i] + m[i]

    with `learning_rate` a positive scaling parameter for the gradient updates
    and `mom_coeff` a value in [0, 1] that determines how much 'friction' there
    is the system and so how quickly previous momentum contributions decay.
    """

    def __init__(self, learning_rate=1e-3, mom_coeff=0.9):
        """Creates a new learning rule object.

        Args:
            learning_rate: A postive scalar to scale gradient updates to the
                parameters by. This needs to be carefully set - if too large
                the learning dynamic will be unstable and may diverge, while
                if set too small learning will proceed very slowly.
            mom_coeff: A scalar in the range [0, 1] inclusive. This determines
                the contribution of the previous momentum value to the value
                after each update. If equal to 0 the momentum is set to exactly
                the negative scaled gradient each update and so this rule
                collapses to standard gradient descent. If equal to 1 the
                momentum will just be decremented by the scaled gradient at
                each update. This is equivalent to simulating the dynamic in
                a frictionless system. Due to energy conservation the loss
                of 'potential energy' as the dynamics moves down the loss
                function surface will lead to an increasingly large 'kinetic
                energy' and so speed, meaning the updates will become
                increasingly large, potentially unstably so. Typically a value
                less than but close to 1 will avoid these issues and cause the
                dynamic to converge to a local minima where the gradients are
                by definition zero.
        """
        super(MomentumLearningRule, self).__init__(learning_rate)
        assert mom_coeff >= 0. and mom_coeff <= 1., (
            'mom_coeff should be in the range [0, 1].'
        )
        self.mom_coeff = mom_coeff

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        This must be called before `update_params` is first called.

        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
        super(MomentumLearningRule, self).initialise(params)
        self.moms = []
        for param in self.params:
            self.moms.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their intial values.

        For this learning rule this corresponds to zeroing all the momenta.
        """
        for mom in zip(self.moms):
            mom *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        for param, mom, grad in zip(self.params, self.moms, grads_wrt_params):
            mom *= self.mom_coeff
            mom -= self.learning_rate * grad
            param += mom
            
class RMSPropLearningRule(GradientDescentLearningRule):
       """
    Root Mean Square propagation.

    Root Mean Square (RMS) propagation protects against vanishing and
    exploding gradients. In RMSprop, the gradient is divided by a running
    average of recent gradients. Given the parameters :math:`\\theta`, gradient :math:`\\nabla J`,
    we keep a running average :math:`\\mu` of the last :math:`1/\\lambda` gradients squared.
    The update equations are then given by

    .. math::

        \\rootmeansquare' &= \\lambda\\rootmeansquare + (1-\\lambda)(\\nabla J)^2

    .. math::

        \\param' &= \\param - \\frac{\\alpha}{\\sqrt{\\rootmeansquare + \\epsilon} + \\epsilon}\\nabla J

    where we use :math:`\\epsilon` as a (small) smoothing factor to prevent from dividing by zero.
    """

    def __init__(self, stochastic_round=False, decay_rate=0.95, learning_rate=2e-3, epsilon=1e-6,
                 gradient_clip_norm=None, gradient_clip_value=None, param_clip_value=None,
                 name=None, schedule=Schedule()):
        """
        Class constructor.

        Arguments:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            decay_rate (float): decay rate of states
            learning_rate (float): the multiplication coefficent of updates
            epsilon (float): smoothing epsilon to avoid divide by zeros
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip
                                                   gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip
                                                parameters.
                                                Defaults to None.
            schedule (neon.optimizers.optimizer.Schedule, optional): Learning rate schedule.
                                                                     Defaults to a constant.
        Notes:
            Only constant learning rate is supported currently.
        """
        super(RMSPropLearningRule, self).__init__(learning_rate)
        self.state_list = None
        self.epsilon = epsilon
        self.decay_rate = decay_rate
        self.learning_rate = learning_rate
        self.schedule = schedule
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
        self.stochastic_round = stochastic_round
        
    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        This must be called before `update_params` is first called.

        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
         """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """

        super(RMSPropLearningRule, self).initialise(params)
        self.rms = []
        for param in self.params:
            self.rms.append(np.zeros_like(param))

    def reset(self):
        """Resets any additional state variables to their intial values.

        For this learning rule this corresponds to zeroing all the momenta.
        """
        for rm in zip(self.rms):
            rm *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        self.rms = self.decay_rate * self.rms + np.square(grads_wrt_params) * (1.0 - self.decay_rate)
        for param, rm, grad in zip(self.params, rms, grads_wrt_params):
            param =  param - (grad*self.learning_rate)/np.sqrt(rm+self.epslion)
            
class AdaMLearningRule(GradientDescentLearningRule):
       """
    Adam optimizer.

    The Adam optimizer combines features from RMSprop and Adagrad. We
    accumulate both the first and second moments of the gradient with decay
    rates :math:`\\beta_1` and :math:`\\beta_2` corresponding to window sizes of
    :math:`1/\\beta_1` and :math:`1/\\beta_2`, respectively.

    .. math::
        m' &= \\beta_1 m + (1-\\beta_1) \\nabla J

    .. math::
        v' &= \\beta_2 v + (1-\\beta_2) (\\nabla J)^2

    We update the parameters by the ratio of the two moments:

    .. math::
        \\theta = \\theta - \\alpha \\frac{\\hat{m}'}{\\sqrt{\\hat{v}'}+\\epsilon}

    where we compute the bias-corrected moments :math:`\\hat{m}'` and :math:`\\hat{v}'` via

    .. math::
        \\hat{m}' &= m'/(1-\\beta_1^t)

    .. math::
        \\hat{v}' &= v'/(1-\\beta_1^t)
        """
def __init__(self, stochastic_round=False, learning_rate=0.001, beta_1=0.9, beta_2=0.999,
                 epsilon=1e-8, gradient_clip_norm=None, gradient_clip_value=None,
                 param_clip_value=None, name="adam"):
        """
        Class constructor.

        Args:
            stochastic_round (bool): Set this to True for stochastic rounding.
                                     If False rounding will be to nearest.
                                     If True will perform stochastic rounding using default width.
                                     Only affects the gpu backend.
            learning_rate (float): the multiplicative coefficient of updates
            beta_1 (float): Adam parameter beta1
            beta_2 (float): Adam parameter beta2
            epsilon (float): numerical stability parameter
            gradient_clip_norm (float, optional): Target gradient norm.
                                                  Defaults to None.
            gradient_clip_value (float, optional): Value to element-wise clip gradients.
                                                   Defaults to None.
            param_clip_value (float, optional): Value to element-wise clip parameters.
                                                Defaults to None.
        """
        super(Adam, self).__init__(learning_rate)
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.stochastic_round = stochastic_round
        self.gradient_clip_norm = gradient_clip_norm
        self.gradient_clip_value = gradient_clip_value
        self.param_clip_value = param_clip_value
        self.t = 0 
        
    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        This must be called before `update_params` is first called.

        Args:
            params: A list of the parameters to be optimised. Note these will
                be updated *in-place* to avoid reallocating arrays on each
                update.
        """
         """
        Apply the learning rule to all the layers and update the states.

        Arguments:
            layer_list (list): a list of Layer objects to optimize.
            epoch (int): the current epoch, needed for the Schedule object.
        """

        super(AdaMLearningRule, self).initialise(params)
        self.adam = []
        for param in self.params:
            self.adam.extend([np.zeros_like(param) for i in range(2)])

    def reset(self):
        """Resets any additional state variables to their intial values.

        For this learning rule this corresponds to zeroing all the momenta.
        """
        for ad in zip(self.adam):
            adam *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters.

        All parameter updates are performed using in-place operations and so
        nothing is returned.

        Args:
            grads_wrt_params: A list of gradients of the scalar loss function
                with respect to each of the parameters passed to `initialise`
                previously, with this list expected to be in the same order.
        """
        self.t = self.t + 1
        l = (self.learning_rate *np.sqrt(1 - self.beta_2 ** self.t) /
             (1 - self.beta_1 ** self.t))
        for param, ad, grad in zip(self.params, adam, grads_wrt_params):
            m, v = ad
            m[:] = m * self.beta_1 + (1. - self.beta_1) * grad
            v[:] = v * self.beta_2 + (1. - self.beta_2) * grad * grad
            
            param = param - (scale_factor * l * m)
                        / (np.sqrt(v) + self.epsilon)