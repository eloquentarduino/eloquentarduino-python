class BoardConfiguration:
    """
    Define a fine-grained configuration for your board
    """
    def __init__(self, model_pattern, label=None, specs=None, **kwargs):
        """
        :param model_pattern:
        :param label:
        :param specs:
        """
        assert isinstance(model_pattern, str), 'model_pattern MUST be a string'
        assert label is None or isinstance(label, str), 'label MUST be None or a string'
        assert specs is None or isinstance(specs, dict), 'specs MUST be None or a dict'
        self.model_pattern = model_pattern
        self.label = label
        self.specs = specs or {}
        self.cli_params = kwargs

    def __str__(self):
        """
        Return a readable representation of the board configuration
        """
        if self.label:
            return self.label
        if len(self.cli_params) == 0:
            return self.model_pattern
        params_string = ','.join(['%s=%s' % (k, str(v)) for k, v in self.cli_params.items()])
        return '%s {%s}' % (self.model_pattern, params_string)

    @property
    def name(self):
        """
        Get name
        """
        return str(self)

    @property
    def cpu_speed(self):
        """
        Get CPU speed
        """
        return self.specs.get('cpu_speed', 0)

    @property
    def cpu_family(self):
        """
        Get chip family
        """
        return self.specs.get('cpu_family', '')