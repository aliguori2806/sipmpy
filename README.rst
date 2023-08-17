sipmpy

=======

Python Framework for SiPM data analysis.

* Requirements

Standard Python libraries, such as pandas, matplotlib, threading and iminuit for the fit procedures.

* Installation

Download the sipmpy package. Let <path> be the string of the absolute path of the sipmpy folder.
In your scripts, you can import sys and add in your code the line 

sys.path.insert(0, "<path>")

otherwise you can add <path> to the environmental variable $PYTHONPATH like this

export PYTHONPATH=$PYTHONPATH:<path>
