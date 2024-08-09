#     PyNetSim: A Python-based Network Simulator for Low-Energy Adaptive Clustering Hierarchy (LEACH) Protocol
#     Copyright (C) 2024  F. Fernando Jurado-Lasso (ffjla@dtu.dk)

#     This program is free software: you can redistribute it and/or modify
#     it under the terms of the GNU General Public License as published by
#     the Free Software Foundation, either version 3 of the License, or
#     (at your option) any later version.

#     This program is distributed in the hope that it will be useful,
#     but WITHOUT ANY WARRANTY; without even the implied warranty of
#     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#     GNU General Public License for more details.

#     You should have received a copy of the GNU General Public License
#     along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import os

from pynetsim.plot.confidence_interval import plot_results, process_results

def main(args):
    """
    Main function

    :param args: Arguments
    :type args: argparse.Namespace

    :return: None
    """
    # Plot the results
    output_folder = args.output
    dfs = process_results(args.input, output_folder)
    plot_results(dfs, output_folder)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", nargs='+', type=str,
                        help="Input directories", required=True)
    parser.add_argument("--output", "-o", type=str,
                        required=True, help="Output folder")
    # Create the output folder if it does not exist
    args = parser.parse_args()
    if not os.path.exists(args.output):
        os.makedirs(args.output)
    main(args)
