import configparser
import json
import sys

class CustomConfigParser(configparser.ConfigParser):
    def getlistfloat(self, section, option, fallback=None, delimiter=','):
        """
        Retrieves a config option and converts it to a list of floats.
        Supports both delimiter-separated strings and JSON arrays.
        """
        try:
            val = self.get(section, option)
        except (configparser.NoSectionError, configparser.NoOptionError):
            if fallback is not None:
                return fallback
            raise

        val = val.strip()

        # Handle JSON array format (e.g., "[1.2, 3.4, 5.6]")
        if val.startswith('[') and val.endswith(']'):
            return [float(x) for x in json.loads(val)]

        # Handle delimited format (e.g., "1.2, 3.4, 5.6")
        return [float(x.strip()) for x in val.split(delimiter) if x.strip()]

    def get_required_option(self, section, option):
        """
        Retrieves an option or exits the program if the option or section is missing.
        """
        try:
            return self.get(section, option)
        except (configparser.NoOptionError, configparser.NoSectionError):
            sys.exit(f"Error: Required option '{option}' not found in section [{section}].")

def print_config(config: dict, title: str = "Running Configuration"):
    """
    Prints the configuration formatted inside a visual border block.
    """
    border = "═" * 45
    print(f"\n{border}")
    print(f" {title.upper()}")
    print(border)
    print(json.dumps(config, indent=4, sort_keys=True))
    print(f"{border}\n")
