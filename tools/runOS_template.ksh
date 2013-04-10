#!/bin/ksh
#################################################################
# FILE: SOS.ksh
#
# PROJECT: Successive Orders of Scattering.
#
# RULE: Command file for a SOS simulation.
#       -> Definition of simulation parameters (-keyword -value).
#       -> Call of the main_SOS.ksh shell.
#    
# This shell is available for the SOS code version 5.0,  
# not for previous ones.          -------------------- 
#
#    
# In order to use this shell, the environment variable RACINE
# has to be previously defined :                       ------
#       RACINE: complete access path to the software directory 
#
#
# AUTHOR: Bruno Lafrance ( CS )
# DATE: 2009 / 11 / 12
#
# MOD:VERSION:4.0: main adjustement for a parameter definition by
#                  couples : -Keyword Value
#################################################################



####################################################################
#                                                                  #
#  Environment variables                                           #
#                                                                  #
####################################################################

##-----------------------------------------------------
# SOS_RACINE : access path to the software directory  
# --> SOS_RACINE/exe contains the compilated codes 
##-----------------------------------------------------
export SOS_RACINE=$RACINE

##-----------------------------------------------------
# SOS_RESULT : access path to the results storage directory 
##-----------------------------------------------------
export SOS_RESULT=$RACINE/SOS_TEST

##----------------------------------------------------------
# SOS_RACINE_FIC : access path to WMO and Shettle&Fenn files 
##----------------------------------------------------------
export SOS_RACINE_FIC=$RACINE/fic

##-----------------------------------------------------
# Storage directory of BRDF and BPDF files 
##-----------------------------------------------------

  ## Storage directory of reflection files to Cox & Munk's Sun glint model
  ##----------------------------------------------------------------------
  export dirSUNGLINT=$SOS_RESULT/SURFACE/GLITTER

  ## Storage directory of reflection files to Roujean's BRDF model
  ##--------------------------------------------------------------
  export dirROUJEAN=$SOS_RESULT/SURFACE/ROUJEAN

  ## Storage directory of reflection files to Rondeaux & Herman's BPDF model
  ##-------------------------------------------------------------------------
  export dirRH=$SOS_RESULT/SURFACE/RH

  ## Storage directory of reflection files to Breon's BPDF model
  ##------------------------------------------------------------
  export dirBREON=$SOS_RESULT/SURFACE/BREON

  ## Storage directory of reflection files to Nadal's BPDF model
  ##------------------------------------------------------------
  export dirNADAL=$SOS_RESULT/SURFACE/NADAL


##-----------------------------------------------------
# Storage directory of Mie aerosols files 
##-----------------------------------------------------
  export dirMIE=$SOS_RESULT/MIE

##-----------------------------------------------------
# Storage directory of log files 
##-----------------------------------------------------
  export dirLOG=$SOS_RESULT/LOG

##----------------------------------------------------------------------------------------------
# Storage directory of simulation results (profiles, aerosols parameters, radiance simulations)
##----------------------------------------------------------------------------------------------
  export dirRESULTS=$SOS_RESULT/SOS



####################################################################
#                                                                  #
#  Parametres de simulation                                        #
#                                                                  #
####################################################################


##################################################################
##################################################################
###            CALL OF THE main_SOS.ksh SHELL
##################################################################
##################################################################
