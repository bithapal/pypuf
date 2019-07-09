from os import getpid
from uuid import UUID
from typing import NamedTuple, Iterable

from numpy.random.mtrand import RandomState
from pypuf.experiments.experiment.base import Experiment
from pypuf.learner.neural_networks.mlp_tf import MultiLayerPerceptronTensorflow
from pypuf.simulation.arbiter_based.ltfarray import LTFArray
from pypuf import tools


class Parameters(NamedTuple):
    seed_simulation: int
    seed_challenges: int
    seed_model: int
    seed_distance: int
    n: int
    k: int
    N: int
    validation_frac: float
    transformation: str
    combiner: str
    preprocessing: str
    layers: Iterable[int]
    activation: str
    loss: str
    domain_in: int
    domain_out: int
    learning_rate: float
    penalty: float
    beta_1: float
    beta_2: float
    tolerance: float
    patience: int
    iteration_limit: int
    batch_size: int
    print_learning: bool


class Result(NamedTuple):
    name: str
    experiment_id: UUID
    pid: int
    accuracy: float
    iterations: int
    measured_time: float
    loss_curve: Iterable[float]
    accuracy_curve: Iterable[float]


class ExperimentMLPTensorflow(Experiment):
    """
    This Experiment uses the MLP learner on an LTFArray PUF simulation.
    """

    NAME = 'Multilayer Perceptron (tensorflow)'
    NUM_ACCURACY = 10000

    def __init__(self, progress_log_prefix, parameters):
        progress_log_name = None if not progress_log_prefix else \
            '{}_MLP_0x{}_0x{}_0_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}'.format(
                progress_log_prefix,
                parameters.seed_model,
                parameters.seed_simulation,
                parameters.n,
                parameters.k,
                parameters.N,
                parameters.validation_frac,
                parameters.layers,
                parameters.activation,
                parameters.loss,
                parameters.domain_in,
                parameters.domain_out,
                parameters.learning_rate,
                parameters.beta_1,
                parameters.beta_2,
                parameters.tolerance,
                parameters.patience,
                parameters.transformation,
                parameters.combiner,
                parameters.preprocessing,
            )
        super().__init__(progress_log_name=progress_log_name, parameters=parameters)
        self.training_set = None
        self.simulation = None
        self.learner = None
        self.model = None

    def prepare(self):
        """
        Prepare learning: initialize learner, prepare training set, etc.
        """
        self.simulation = LTFArray(
            weight_array=LTFArray.normal_weights(
                self.parameters.n,
                self.parameters.k,
                random_instance=RandomState(seed=self.parameters.seed_simulation),
            ),
            transform=self.parameters.transformation,
            combiner=self.parameters.combiner,
        )
        prng_challenges = RandomState(seed=self.parameters.seed_challenges)
        N_val = int(self.parameters.N * self.parameters.validation_frac)
        N_train = self.parameters.N - N_val
        self.training_set = tools.TrainingSet(self.simulation, N_train, prng_challenges)
        validation_set = tools.TrainingSet(self.simulation, N_val, prng_challenges)
        self.learner = MultiLayerPerceptronTensorflow(
            n=self.parameters.n,
            k=self.parameters.k,
            training_set=self.training_set,
            validation_set=validation_set,
            transformation=self.simulation.transform,
            preprocessing=self.parameters.preprocessing,
            layers=self.parameters.layers,
            activation=self.parameters.activation,
            loss=self.parameters.loss,
            domain_in=self.parameters.domain_in,
            domain_out=self.parameters.domain_out,
            learning_rate=self.parameters.learning_rate,
            penalty=self.parameters.penalty,
            beta_1=self.parameters.beta_1,
            beta_2=self.parameters.beta_2,
            tolerance=self.parameters.tolerance,
            patience=self.parameters.patience,
            iteration_limit=self.parameters.iteration_limit,
            batch_size=self.parameters.batch_size,
            seed_model=self.parameters.seed_model,
            print_learning=self.parameters.print_learning,
        )
        self.learner.prepare()

    def run(self):
        """
        Train the learner.
        """
        self.model = self.learner.learn()

    def analyze(self):
        """
        Calculate statistics of the trained model and write them into a result log file.
        """
        assert self.model is not None
        accuracy = 1.0 - tools.approx_dist(
                instance1=self.simulation,
                instance2=self.model,
                num=min(self.NUM_ACCURACY, 2 ** self.parameters.n),
                random_instance=RandomState(seed=self.parameters.seed_distance),
        )
        return Result(
            name=self.NAME,
            experiment_id=self.id,
            pid=getpid(),
            accuracy=accuracy,
            iterations=self.learner.history.epoch[-1],
            measured_time=self.measured_time,
            loss_curve=self.learner.history.history['val_loss'],
            accuracy_curve=self.learner.history.history['val_pypuf_accuracy'],
        )
