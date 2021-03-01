import haiku as hk
import jax.numpy as jnp
import jax
from jax.scipy.special import logsumexp
from typing import Optional, Callable, Tuple
import jax.lax as lax

class crf_layer(hk.Module):

    def __init__(self, 
                 n_classes: int, 
                 transition_init: Optional[hk.initializers.Initializer] = jnp.ones,
                 name: Optional[str] = None):
        """Constructs a simple CRF layer.

        Args:
            n_classes: The total number of possible labels of any token in the input sequence.
            transition_init: The initializer to use for initializing the transition matrix of 
                            size [n_classes, n_classes]. By default, uses all ones to initialize.
                            The (i,j)-th entry contains the score for transitioning from j to i.
            name: Name of the module
        """
        super().__init__(name=name)
        self.n_classes = n_classes
        self.transition_init = transition_init
        self.init_alphas = jnp.full((self.n_classes,), 0.0)

    def core_recursion(self, fn: Callable,
                       transition_matrix: jnp.ndarray,
                       prev_alphas: jnp.ndarray, 
                       logit_t: jnp.ndarray) -> jnp.ndarray:
        """The core recursion used in CRF
        Args:
            fn: A function that accumulates the results of transitioning to a particular class
                at t-th time step from any of the classes at the previous time step, into a single number.
            transition_matrix: The transition matrix of CRF of size [n_classes, n_classes].
                               The (i,j)-th entry contains the score for transitioning from j to i.
            prev_alphas: A tensor of size [n_classes,] that is being computed recursively using the DP of CRF.
            logit_t: The emission scores at a time step t for each class. A tensor of size [n_classes,]
        
        Returns:
            A tensor of size [n_classes,] where the i-th entry is the accumulation of all scores of all sequences
            ending at the i-th tag at time step t.
        """
        prev_alphas = fn( jnp.expand_dims(logit_t/self.n_classes, 1)
                          +transition_matrix+prev_alphas, axis=1 )
        
        return prev_alphas
    
    def sum_scores(self,
                   transition_matrix: jnp.ndarray,
                   logits: jnp.ndarray,) -> jnp.ndarray:
        """Sums together the scores of all possible sequences
        Args:
            transition_matrix: The transition matrix of CRF of size [n_classes, n_classes].
                               The (i,j)-th entry contains the score for transitioning from j to i.
            logits: The emission scores (of each class) for a sequence inputs of size [T, n_classes]
        
        Returns:
            A tensor of size [T,] where the entry at i-th index contains the sum of 
            scores of all sequences of length (i+1).
        """
        scan_fn = lambda prev_alphas, logit_t: (self.core_recursion(logsumexp, transition_matrix, prev_alphas, logit_t),)*2
        alphas = lax.scan(scan_fn, init=self.init_alphas, xs=logits)
        return jnp.sum(alphas[1], -1)
    
    def score_sequence(self,
                       transition_matrix: jnp.ndarray,
                       logits: jnp.ndarray,
                       tags: jnp.ndarray) -> jnp.ndarray:
        """Calculates the score of a given sequence
        Args:
            transition_matrix: The transition matrix of CRF of size [n_classes, n_classes]
                               The (i,j)-th entry contains the score for transitioning from j to i.
            logits: The emission scores (of each class) for a sequence inputs of size [T, n_classes]
            tags: The tags(class label) at each position. A matrix of size [T,]
        
        Returns:
            A tensor of size [T,] where the i-th entry contains the score of sequence of the first (i+1) tags.
        """
        first_tag_score = logits[0, tags[0]]+jnp.sum(transition_matrix[tags[0],:])
        scan_fn = lambda prev_score, i: (prev_score + transition_matrix[tags[i+1]][tags[i]] + logits[i+1, tags[i+1]],)*2
        final_scores = lax.scan(scan_fn, init=first_tag_score, xs=jnp.arange(logits.shape[0]-1))
        return final_scores[1]

    def viterbi_decode(self,
                       transition_matrix: jnp.ndarray,
                       logits: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implements the Viterbi decoding algorithm for the CRF.
        Args:
            transition_matrix: The transition matrix of CRF of size [n_classes, n_classes]
                               The (i,j)-th entry contains the score for transitioning from j to i.
            logits: The emission scores (of each class) for a sequence inputs of size [T, n_classes]
        
        Returns:
            A tuple where the 
            first element is: a tensor of size [T, n_classes] where the (t,i)-th entry is the maximum score
                              that can be attained by a tag sequence of length (t+1) ending at class label i
            second element is: a tensor of size [T, n_classes] where the (t,i)-th entry is the class label at 
                               t-th position in the max. scoring sequence of length (t+1) ending at class label i.
                               The 0-th entry of this tensor is to be discarded away.
        """
        prev_alphas = self.init_alphas

        scores_lis = []
        tags_lis = []
        
        for i in range(logits.shape[0]):
            max_tags = self.core_recursion(jnp.argmax, transition_matrix, prev_alphas, logits[i])
            prev_alphas = self.core_recursion(jnp.max, transition_matrix, prev_alphas, logits[i])
            scores_lis.append(prev_alphas)
            tags_lis.append(max_tags)
        
        return jnp.stack(scores_lis), jnp.stack(tags_lis)

    
    def batched_sum_scores(self,
                           batch_logits: jnp.ndarray,
                           lengths: jnp.ndarray) -> jnp.ndarray:
        """Implements batched version of sum_score function
        Args:
            batch_logits: Emission scores of size [N,T,n_classes] where N is batch size
            lengths: length of each element in the batch. A tensor of size [N,]
        Returns:
            Tensor of size [N,] where the i-th entry corresponds to the sum of scores of 
            possible sequences of length lengths[i] over the i-th sample in the batch.
        """
        transition_matrix = hk.get_parameter("transition_matrix", 
                                             [self.n_classes, self.n_classes], 
                                             init=self.transition_init)
        
        batch_score_fn = jax.vmap(lambda logits: self.sum_scores(transition_matrix, logits),
                                  in_axes=(0,), out_axes=0)
        
        sum_scores = batch_score_fn(batch_logits)

        return jnp.diag(sum_scores[:,lengths-1])
    
    def batched_score_sequence(self,
                               batch_logits: jnp.ndarray,
                               lengths: jnp.ndarray, 
                               batch_tags: jnp.ndarray) -> jnp.ndarray:
        """Implements batched version of score_sequence function
        Args:
            batch_logits: Emission scores of size [N,T,n_classes] where N is batch size
            lengths: length of each element in the batch. A tensor of size [N,]
            batch_tags: tags/class labels for each position of the N samples. A tensor of size [N,T]
        Returns:
            A tensor of size [N,], where the i-th element is the score of i-th sequence of batch_tags.
        """
        transition_matrix = hk.get_parameter("transition_matrix", 
                                             [self.n_classes, self.n_classes], 
                                             init=self.transition_init)
        
        batch_seq_score_fn = jax.vmap(lambda logits, tags: self.score_sequence(transition_matrix, logits, tags),
                                      in_axes=(0,0), out_axes=0)

        seq_scores = batch_seq_score_fn(batch_logits, batch_tags)
        return jnp.diag(seq_scores[:,lengths-1])
    
    def batch_viterbi_decode(self,
                             batch_logits: jnp.ndarray,
                             lengths: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Implements batched version of viterbi_decode function
        Args:
            batch_logits: Emission scores of size [N,T,n_classes] where N is batch size
            lengths: length of each element in the batch. A tensor of size [N,]
        Returns:
            A tuple where the 
            first element is: A tensor of size [N,T] having the max. scoring tag sequences 
                              for each element in the batch, padded with -1.
            second element is: A tensor of size [N,] having maximum scores(over all possible tag sequences)
                               ,for each sample in the batch.
        """
        transition_matrix = hk.get_parameter("transition_matrix", 
                                             [self.n_classes, self.n_classes], 
                                             init=self.transition_init)
        
        batch_decode_fn = jax.vmap(lambda logits: self.viterbi_decode(transition_matrix, logits),
                                   in_axes=(0), out_axes=(0,0))
        
        scores, tags = batch_decode_fn(batch_logits)
        
        tag_sequences = [jnp.array([-1]*scores.shape[0])]
        batch_scores = jnp.arary([-1]*scores.shape[0])
        
        for i in range(scores.shape[1], 0, -1):
            
            last_tag = jnp.where(i==lengths, jnp.argmax(scores[:,i,:], axis=1), -1)
            tag_sequences.append( jnp.diag(jnp.where(i<lengths, tags[:,i,tag_sequences[-1]], last_tag)) )
            batch_scores = jnp.where(i==lengths, jnp.max(scores[:,i,:], axis=1), batch_scores)
        
        return jnp.stack(reverse(tag_sequences), axis=1), batch_scores
    
    def __call__(self,
                 batch_logits: jnp.ndarray,
                 lengths: jnp.ndarray, 
                 batch_tags: jnp.ndarray) -> jnp.ndarray:
        """Computes the negative log likelihood of the sequences provided 
        in batch_tags, under the transition matrix of CRF and the emission probabilities specified
        by logits.
        Args:
            batch_logits: Emission scores of size [N,T,n_classes] where N is batch size
            lengths: length of each element in the batch. A tensor of size [N,]
            batch_tags: tags/class labels for each position of the N samples. A tensor of size [N,T]
        Returns:
            A tensor of size [N,] having the negative log likelihood of each sample in the batch.            
        """
        partition_fn = self.batched_sum_scores(batch_logits, lengths)
        gold_score = self.batched_score_sequence(batch_logits, lengths, batch_tags)
        return partition_fn-gold_score
