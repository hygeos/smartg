aer_URL_old = https://docs.hygeos.com/s/F5L9t86DfAiPBBB/download
aer_URL = https://docs.hygeos.com/s/8PnKXFXQbmYyTte/download
acs_URL = https://docs.hygeos.com/s/HwotAHPstdCCKcJ/download
atm_URL = https://docs.hygeos.com/s/z6MRf9g66WmWeBA/download
STP_URL = https://docs.hygeos.com/s/NW42DNPtKw3NNW7/download
valid_URL = https://docs.hygeos.com/s/6EPBqwebn94NYPq/download
water_URL = https://docs.hygeos.com/s/3NKP5tMsHKnNRpt/download
WGET    = @wget -c -P


auxdata_all: auxdata_aerosols_old auxdata_aerosols auxdata_acs auxdata_atm \
 auxdata_STP auxdata_valid auxdata_water

auxdata_aerosols_old:
	$(WGET) auxdata/ $(aer_URL_old)/aerosols.zip
	unzip auxdata/aerosols.zip -d auxdata/
	rm -f auxdata/aerosols.zip
	mv auxdata/aerosols auxdata/aerosols_old

auxdata_aerosols:
	$(WGET) auxdata/ $(aer_URL)/aerosols.zip
	unzip auxdata/aerosols.zip -d auxdata/
	rm -f auxdata/aerosols.zip

auxdata_acs:
	$(WGET) auxdata/ $(acs_URL)/acs.zip
	unzip auxdata/acs.zip -d auxdata/
	rm -f auxdata/acs.zip

auxdata_atm:
	$(WGET) auxdata/ $(atm_URL)/atmospheres.zip
	unzip auxdata/atmospheres.zip -d auxdata/
	rm -f auxdata/atmospheres.zip

auxdata_STP:
	$(WGET) auxdata/ $(STP_URL)/STPs.zip
	unzip auxdata/STPs.zip -d auxdata/
	rm -f auxdata/STPs.zip

auxdata_valid:
	$(WGET) auxdata/ $(valid_URL)/validation.zip
	unzip auxdata/validation.zip -d auxdata/
	rm -f auxdata/validation.zip

auxdata_water:
	$(WGET) auxdata/ $(water_URL)/water.zip
	unzip auxdata/water.zip -d auxdata/
	rm -f auxdata/water.zip

.PHONY: clean
clean:
	rm -rf auxdata/aerosols_old
	rm -rf auxdata/aerosols
	rm -rf auxdata/acs
	rm -rf auxdata/atmospheres
	rm -rf auxdata/STPs
	rm -rf auxdata/validation
	rm -rf auxdata/water
