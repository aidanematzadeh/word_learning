import ConfigParser
import sys
"""
learnconfig.py

Extension to generic Python ConfigParser for parsing a config file appropriate
for the Learner class of learn.py.

"""


class LearnerConfig(object):
    """
    Encapsulated ConfigPaser specific to a Learner.
    See config.ini for the settings that this is expected to parse.
    
    Members:
        config_params -- dictionary of parameter names and values
    
    """

    def __init__(self, config_path):
        """ Create a LearnerConfig from the file located at config_path. """
        self._config_params = {}
        
        config = ConfigParser.ConfigParser()
        config.read(config_path)
        
        for section in config.sections():
            for option in config.options(section):
                self._config_params[option] = config.get(section, option)

        if len(self._config_params) < 1:
            print "ERROR: config file not found: " + config_path
            sys.exit(2)
        
    def param(self, param):
        """ 
        Return the string value of parameter param if it exists, otherwise 
        return the empty string.
        
        """
        if param in self._config_params:
            return self._config_params[param]
        else:
            return ''
        
    def param_int(self, param):
        """
        Return the integer value of parameter param if it exists, otherwise 
        return -1.
        
        """
        if param in self._config_params:
            try:
                return int(self._config_params[param])
            except:
                p = self._config_params[param]
                print "ERROR [Config File]: Parameter %s is not a valid int" % p
                sys.exit(2)
        else:
            return -1
        
    def param_float(self, param):
        """
        Return the float value of parameter param if it exists, otherwise return
        -1.0.
        
        """
        if param in self._config_params:
            try:
                return float(self._config_params[param])
            except:
                p = self._config_params[param]
                print "ERROR [Config File]: Parameter %s is not a valid float" % p
                sys.exit(2)
        else:
            return -1.0
        
    def param_bool(self, param):
        """
        Return the boolean value of parameter param if it exists, otherwise 
        return false.
        
        """
        if param in self._config_params:
            return self._config_params[param].lower() == 'true'
        else:
            return False
    
    def has_param(self, param):
        """ Return whether param exists or not in this config. """
        return param in self._config_params
            
                
