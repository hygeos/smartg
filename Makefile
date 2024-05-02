aer_URL = https://docs.hygeos.com/s/F5L9t86DfAiPBBB/download
aer_URL_new = https://docs.hygeos.com/s/8PnKXFXQbmYyTte/download
acs_URL = https://docs.hygeos.com/s/HwotAHPstdCCKcJ/download
WGET    = @wget -c -P

auxdata_aerosols:
	$(WGET) auxdata/ $(aer_URL)/aerosols.zip
	unzip auxdata/aerosols.zip -d auxdata/
	rm -f auxdata/aerosols.zip

auxdata_aerosols_new:
	$(WGET) auxdata/new_aer/ $(aer_URL_new)/aerosols.zip
	unzip auxdata/new_aer/aerosols.zip -d auxdata/new_aer/
	rm -f auxdata/new_aer/aerosols.zip

auxdata_acs:
	$(WGET) auxdata/ $(acs_URL)/acs.zip
	unzip auxdata/acs.zip -d auxdata/
	rm -f auxdata/acs.zip

.PHONY: clean
clean:
	rm -rf auxdata/aerosols
	rm -rf auxdata/new_aer
	rm -rf auxdata/acs
