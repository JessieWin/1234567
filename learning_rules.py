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
    def  __init__ (self, learning_rate, decay_rate=0.1):
        """Creates a new learning rule object."""
        super(RMSPropLearningRule, self).__init__(learning_rate)
        assert decay_rate >= 0. and decay_rate <= 1., ('Decay rate should be in the range [0, 1].')
        self.decay_rate = decay_rate
        self.epsilon = 0.00001

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.

        For RMSProp, just set the accumulated mean square of gradients to be 0
        """
        super(RMSPropLearningRule, self).initialise(params)
        meansquare = []
        for param in self.params:
            meansquare.append(np.zeros_like(param))
        self.meansquare = np.array(meansquare)

    def reset(self):
        """Resets any additional state variables to their intial values."""
        for ms in zip(self.meansquare):
            ms *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters."""
        grads = np.array(grads_wrt_params)
        self.meansquare = self.decay_rate * (grads**2) + (1 - self.decay_rate) * self.meansquare
        for ms, param, grad in zip(self.meansquare, self.params, grads):
            #ms = self.decay_rate * grad**2 + (1 - self.decay_rate) * ms
            delta = - grad * self.learning_rate / (np.sqrt(ms)+self.epsilon)
            param += delta

class AdaMLearningRule(GradientDescentLearningRule):
    
    def  __init__ (self, learning_rate, beta1=0.9, beta2=0.99):
        """Creates a new learning rule object.
        """
        super(AdaMLearningRule, self).__init__(learning_rate)
        
        assert beta1 >= 0. and beta1 <= 1., ('1st moment decay rate should be in the range [0, 1].')
        self.beta1 = beta1
        assert beta2 >= 0. and beta2 <= 1., ('2nd moment decay rate should be in the range [0, 1].')
        self.beta2 = beta2
        self.epsilon = 1e-8
        self.timestamp = 0

    def initialise(self, params):
        """Initialises the state of the learning rule for a set or parameters.
        """
        super(AdaMLearningRule, self).initialise(params)
        firstmoment = []
        secondmoment = []
        for param in self.params:
            firstmoment.append(np.zeros_like(param))
            secondmoment.append(np.zeros_like(param))
        self.firstmoment = np.array(firstmoment)
        self.secondmoment = np.array(secondmoment)

    def reset(self):
        """Resets any additional state variables to their intial values.
        """
        self.timestamp = 0
        for fmom, smom in zip(self.firstmoment, self.secondmoment):
            fmom *= 0.
            smom *= 0.

    def update_params(self, grads_wrt_params):
        """Applies a single update to all parameters.
        """
        grads_wrt_params = np.array(grads_wrt_params)
        self.timestamp += 1
        self.firstmoment = self.firstmoment * self.beta1 + (1. - self.beta1) * grads_wrt_params
        self.secondmoment = self.secondmoment * self.beta2 + (1. - self.beta2) * (grads_wrt_params**2)
        for fm, sm, param in zip(self.firstmoment, self.secondmoment, self.params):
            firsthat = fm/(1-self.beta1**self.timestamp)
            secondhat = sm/(1-self.beta2**self.timestamp)
            param -=  self.learning_rate * firsthat/(np.sqrt(secondhat)+self.epsilon)