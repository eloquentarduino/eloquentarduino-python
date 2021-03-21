class GridSearch:
    """
    Abstract base for GridSearch implementations
    """
    def __init__(self):
        self.results = []
        self.classifier_constraints = []
        self.resources_contraints = []
        self.runtime_constraints = []

    def min_accuracy(self, accuracy):
        """
        Add min accuracy constraint
        :param accuracy: float
        :return: self
        """
        return self.add_classifier_constraint(lambda result: result.accuracy >= accuracy)

    def max_rel_flash(self, flash):
        """
        Add maximum relative flash constraint
        :param flash: int
        :return: self
        """
        return self.add_resource_constraint(lambda result: result.resources['rel_flash'] <= flash)

    def max_rel_memory(self, memory):
        """
        Add maximum relative memory constraint
        :param memory: int
        :return: self
        """
        return self.add_resource_constraint(lambda result: result.resources['rel_memory'] <= memory)

    def max_rel_flash_percent(self, flash):
        """
        Add maximum relative flash constraint
        :param flash: float
        :return: self
        """
        return self.add_resource_constraint(lambda result: result.resources['rel_flash_percent'] <= flash)

    def max_rel_memory_percent(self, memory):
        """
        Add maximum relative memory constraint
        :param memory: float
        :return: self
        """
        return self.add_resource_constraint(lambda result: result.resources['rel_memory_percent'] <= memory)

    def max_inference_time(self, inference_time):
        """
        Add maximum relative memory constraint
        :param inference_time: float
        :return: self
        """
        return self.add_runtime_constraint(lambda result: result.inference_time <= inference_time)

    def add_classifier_constraint(self, constraint):
        """
        Add constraint on classifier
        :param constraint: callable
        :return: self
        """
        assert callable(constraint), 'constraint MUST be callable'
        self.classifier_constraints.append(constraint)

        return self

    def add_resource_constraint(self, constraint):
        """
        Add constraint on resources
        :param constraint: callable
        :return: self
        """
        assert callable(constraint), 'constraint MUST be callable'
        self.resources_contraints.append(constraint)

        return self

    def add_runtime_constraint(self, constraint):
        """
        Add constraint on on-device execution
        :param constraint: callable
        :return: self
        """
        assert callable(constraint), 'constraint MUST be callable'
        self.runtime_constraints.append(constraint)

        return self

    def collect_resources(self):
        """
        Turn on the collection of resources, without constraints
        :return: self
        """
        # add a fake constraint to force the collection of resources
        return self.max_rel_flash_percent(1)

    def collect_inference_time(self):
        """
        Turn on the collection of inference time, without constraints
        :return: self
        """
        # add a fake constraint to force the collection of inference time
        return self.max_inference_time(10e10)

    def has_resource_constraints(self):
        """
        Test if any constraint on resources has been set
        :return: bool
        """
        return len(self.resources_contraints) > 0

    def has_runtime_constraints(self):
        """
        Test if any constraint on runtime has been set
        :return: bool
        """
        return len(self.runtime_constraints) > 0

    def passes_classifier_constraints(self, result):
        """
        Test if a result passes classifier constraints
        :param result: GridSearchResult
        :return: bool
        """
        return self.passes_constraints(self.classifier_constraints, result)

    def passes_resources_constraints(self, result):
        """
        Test if a result passes resources constraints
        :param result: GridSearchResult
        :return: bool
        """
        return self.passes_constraints(self.resources_contraints, result)

    def passes_runtime_constraints(self, result):
        """
        Test if a result passes runtime constraints
        :param result: GridSearchResult
        :return: bool
        """
        return self.passes_constraints(self.runtime_constraints, result)

    def passes_constraints(self, constraints, result):
        """
        Test if a result passes all the constraints
        :param result: GridSearchResult
        :return: bool
        """
        for constraint in constraints:
            if not constraint(result):
                return False

        return True

    def append_result(self, result, project):
        """
        Append result if it passes all constraints
        :param result: GridSearchResult
        """
        if not self.passes_classifier_constraints(result):
            return

        if self.has_resource_constraints():
            result.resources = result.clf.on_device(project=project).get_resources()

            if not self.passes_resources_constraints(result):
                return

            if self.has_runtime_constraints():
                result.inference_time = result.clf.on_device(project).get_inference_time()

                if not self.passes_runtime_constraints(result):
                    return

        self.results.append(result)
