# NeMo-RL distillation

Domain language for policy distillation and training/generation consistency.

## Language

**Student policy**:
The trainable policy whose response distribution is being adapted using teacher feedback.

**Teacher policy**:
A fixed policy that defines a target distribution for student responses.

**Multi-Teacher On-Policy Distillation (MOPD)**:
Distillation on student-generated responses with the teacher selected by the response's task or source group.
_Avoid_: Teacher ensemble, when each response is assigned to one teacher.

**Teacher routing**:
The assignment of a response's task or source group to the teacher that supplies its target distribution.

**Full-vocabulary distillation**:
Distillation that evaluates its divergence across the complete next-token vocabulary at each evaluated context.
_Avoid_: Full-data training, which describes the prompt pool rather than the vocabulary used by the objective.

**Sampled-token logprob gap**:
The difference between student and teacher log probabilities for a sampled response token. It is distinct from a divergence explicitly summed over the vocabulary.

**Training/generation mismatch**:
A difference between training-side and generation-side model probabilities for the same weights, tokenized context, and distribution settings.
_Avoid_: Policy staleness, which changes the model weights being compared.

**Policy staleness**:
The difference in policy version between response generation and the training update that consumes that response.

**Self-distillation control**:
A distillation comparison in which the student and teacher start from identical weights, providing a reference for numerical and data-path correctness.
