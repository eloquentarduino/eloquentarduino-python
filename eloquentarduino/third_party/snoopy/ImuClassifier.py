from eloquentarduino.utils import jinja


class ImuClassifier:
    """
    Third-party utility class to generate a IMU classifier
    @added 0.1.19
    """
    def __init__(self, class_name='ML'):
        """
        :param class_name: str
        """
        self.template_data = {
            'class_name': class_name,
            'persist_prediction': True,
            'debug_enabled': True,
            'extern_variable': 'mlPredictionByte',
            'features': 'yaw, pitch, roll, Ax / aRes, Ay / aRes, Az / aRes',
            'pipeline_filename': 'Pipeline',
            'pipeline_variable': 'pipeline'
        }

    def persist_prediction(self, enabled=True):
        """
        Toggle persistent prediction
        :param enabled: bool
        """
        self.template_data['persist_prediction'] = enabled

        return self

    def debug(self, enabled=True):
        """
        Toggle debugging messages
        :param enabled: bool
        """
        self.template_data['debug_enabled'] = enabled

        return self

    def extern_variable(self, variable_name):
        """
        Set extern variable name
        :param variable_name: str
        """
        self.template_data['extern_variable'] = variable_name

        return self

    def set_features(self, features):
        """
        Set feature vector
        :param features: str
        """
        self.template_data['features'] = features

        return self

    def set_pipeline_filename(self, filename):
        """
        Set external pipeline file name
        :param filename: str
        """
        self.template_data['pipeline_filename'] = filename

        return self

    def set_pipeline_variable_name(self, variable_name):
        """
        Set external pipeline singleton name
        :param variable_name: str
        """
        self.template_data['pipeline_variable'] = variable_name

        return self

    def port(self):
        """
        Port to C++
        """
        return jinja('third_party/snoopy/ImuClassifier.jinja', template_data=self.template_data)