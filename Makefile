aer_URL = https://docs.hygeos.com/s/F5L9t86DfAiPBBB/download
WGET    = @wget -c -P

auxdata_aerosols:
	$(WGET) auxdata/ $(aer_URL)/aerosols.zip
	unzip auxdata/aerosols.zip -d auxdata/
	rm -f auxdata/aerosols.zip

.PHONY: clean
clean:
	rm -rf auxdata/aerosols
