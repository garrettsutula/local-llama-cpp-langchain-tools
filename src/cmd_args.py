import argparse, os
from dotenv import dotenv_values

parser = argparse.ArgumentParser(description="SD.Next", conflict_handler='resolve', epilog='For other options see UI Settings page', prog='', add_help=True, formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=55, indent_increment=2, width=200))
parser._optionals = parser.add_argument_group('Other options') # pylint: disable=protected-access
group = parser.add_argument_group('cli options')

group.add_argument("-n", "--name", type=str, help="Job file name to load and run from './jobs/'")

def get_args():
    args = parser.parse_args()

    return args