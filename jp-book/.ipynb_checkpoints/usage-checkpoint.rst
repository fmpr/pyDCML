{
 "cells": [
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "# Basic usage\n",
    "\n",
    "Specifying choice models using the formula interface provided by Dan is very simple. We start by defining the fixed effects parameters and the random effects parameters:"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": [
    "# define fixed effects parameters\n",
    "B_XF1 = FixedEffect('BETA_XF1')\n",
    "B_XF2 = FixedEffect('BETA_XF2')\n",
    "B_XF3 = FixedEffect('BETA_XF3')\n",
    "\n",
    "# define random effects parameters\n",
    "B_XR1 = RandomEffect('BETA_XR1')\n",
    "B_XR2 = RandomEffect('BETA_XR2')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "markdown",
   "metadata": {},
   "source": [
    "You can also create content with Jupyter Notebooks. This means that you can include\n",
    "code blocks and their outputs in your book.\n",
    "\n",
    "## Markdown + notebooks\n",
    "\n",
    "As it is markdown, you can embed images, HTML, etc into your posts!\n",
    "\n",
    "![](https://myst-parser.readthedocs.io/en/latest/_static/logo-wide.svg)\n",
    "\n",
    "You can also $add_{math}$ and\n",
    "\n",
    "$$\n",
    "math^{blocks}\n",
    "$$\n",
    "\n",
    "or\n",
    "\n",
    "$$\n",
    "\\begin{aligned}\n",
    "\\mbox{mean} la_{tex} \\\\ \\\\\n",
    "math blocks\n",
    "\\end{aligned}\n",
    "$$\n",
    "\n",
    "But make sure you \\$Escape \\$your \\$dollar signs \\$you want to keep!\n",
    "\n",
    "## MyST markdown\n",
    "\n",
    "MyST markdown works in Jupyter Notebooks as well. For more information about MyST markdown, check\n",
    "out [the MyST guide in Jupyter Book](https://jupyterbook.org/content/myst.html),\n",
    "or see [the MyST markdown documentation](https://myst-parser.readthedocs.io/en/latest/).\n",
    "\n",
    "## Code blocks and outputs\n",
    "\n",
    "Jupyter Book will also embed your code blocks and output in your book.\n",
    "For example, here's some sample Matplotlib code:"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.10"
  },
  "widgets": {
   "application/vnd.jupyter.widget-state+json": {
    "state": {},
    "version_major": 2,
    "version_minor": 0
   }
  }
 },
 "nbformat": 4,
 "nbformat_minor": 4
}
