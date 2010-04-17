#!/bin/bash
#
# Downloads all external dependencies of cl-gpu.
#

cd `dirname $0`

function fetch_darcs() {
	REPO=$1
	DIR=`basename $REPO`

	if [ -d "$DIR" ]; then
		pushd $DIR;
		darcs pull -a
		popd
	else
		darcs get --lazy "$REPO"
	fi
	
	ln -sf $DIR/*.asd .
}

function fetch_wget() {
	SYSTEM=$1
	URL=$2

	if [ ! -e $SYSTEM.asd ]; then
		FILE=`basename $URL`
		wget $URL
		tar -xvzf $FILE
		ln -sf $SYSTEM*/*.asd .
	fi
}

function fetch_git() {
	REPO=$1
	DIR=`basename $REPO .git`
	if [ -d "$DIR" ]; then
		pushd $DIR
		git pull
		popd
	else
		git clone "$REPO"
	fi
	
	ln -sf $DIR/*.asd .
}

wget -N http://common-lisp.net/project/asdf/asdf.lisp

# Better REPL interface for ecl
fetch_wget ecl-readline http://www.common-lisp.net/project/ecl-readline/releases/ecl-readline-0.4.1.tar.gz

# Misc
fetch_wget anaphora http://common-lisp.net/project/anaphora/files/anaphora-latest.tar.gz
fetch_git git://common-lisp.net/projects/alexandria/alexandria.git
fetch_darcs http://common-lisp.net/project/iterate/darcs/iterate
fetch_darcs http://common-lisp.net/project/metabang-bind
#fetch_darcs http://common-lisp.net/project/trivial-shell
fetch_darcs http://common-lisp.net/~loliveira/darcs/trivial-garbage/

# hu.dwim.util (formerly)
#fetch_wget cl-fad http://weitz.de/files/cl-fad.tar.gz

# CFFI
fetch_wget trivial-features http://common-lisp.net/~loliveira/tarballs/trivial-features/trivial-features_latest.tar.gz
fetch_wget babel http://common-lisp.net/project/babel/releases/babel_latest.tar.gz
fetch_wget cffi http://common-lisp.net/project/cffi/releases/cffi_latest.tar.gz

# ContextL
fetch_darcs http://common-lisp.net/project/closer/repos/lw-compat
fetch_darcs http://common-lisp.net/project/closer/repos/closer-mop
fetch_darcs http://common-lisp.net/project/closer/repos/contextl

# Threads
fetch_darcs http://common-lisp.net/project/bordeaux-threads/darcs/bordeaux-threads

# Walker
fetch_darcs http://dwim.hu/darcs/hu.dwim.asdf
fetch_darcs http://dwim.hu/darcs/hu.dwim.common
fetch_darcs http://dwim.hu/darcs/hu.dwim.common-lisp
fetch_darcs http://dwim.hu/darcs/hu.dwim.def
fetch_darcs http://dwim.hu/darcs/hu.dwim.defclass-star
fetch_darcs http://dwim.hu/darcs/hu.dwim.stefil
fetch_darcs http://dwim.hu/darcs/hu.dwim.syntax-sugar
fetch_darcs http://dwim.hu/darcs/hu.dwim.util
fetch_darcs http://dwim.hu/darcs/hu.dwim.walker
