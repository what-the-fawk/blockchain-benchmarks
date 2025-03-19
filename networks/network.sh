# !/bin/bash

# Check reqs
check_required_env_vars() {
    local required_env_vars=("$@")

    for var in "${required_env_vars[@]}"; do
        if [ -z "${!var}" ]; then
            echo "Error: Environment variable $var is not set."
            exit 1
        fi
    done
}

list_envs() {
    local prefix=$1
    local suffix=$2
    local env_vars=()
    for var in $(compgen -e); do
        if [[ -n $prefix && $var == ${prefix}* ]] && [[ -z $suffix || $var == *${suffix} ]]; then
            env_vars+=("$var=${!var}")
        elif [[ -z $prefix && $var == *${suffix} ]]; then
            env_vars+=("$var=${!var}")
        fi
    done
    echo "${env_vars[@]}"
}

check_requirements() {
    check_required_env_vars "FABRIC_CFG" ""

    ## Check if your have cloned the peer binaries and configuration files.
    peer version > /dev/null 2>&1

    if [[ $? -ne 0 || ! -d "./config" ]]; then
        errorln "Config folder not found"
        errorln
        exit 1
    fi

    # use the fabric peer container to see if the samples and binaries match your
    # docker images
    LOCAL_VERSION=$(peer version | sed -ne 's/^ Version: //p')
    DOCKER_IMAGE_VERSION=$(${CONTAINER_CLI} run --rm hyperledger/fabric-peer:latest peer version | sed -ne 's/^ Version: //p')

    infoln "LOCAL_VERSION=$LOCAL_VERSION"
    infoln "DOCKER_IMAGE_VERSION=$DOCKER_IMAGE_VERSION"

    if [ "$LOCAL_VERSION" != "$DOCKER_IMAGE_VERSION" ]; then
        warnln "Local fabric binaries and docker images are out of sync. This may cause problems."
    fi

    for UNSUPPORTED_VERSION in $NONWORKING_VERSIONS; do
        infoln "$LOCAL_VERSION" | grep -q $UNSUPPORTED_VERSION
        if [ $? -eq 0 ]; then
        fatalln "Local Fabric binary version of $LOCAL_VERSION does not match the versions supported by the test network."
        fi

        infoln "$DOCKER_IMAGE_VERSION" | grep -q $UNSUPPORTED_VERSION
        if [ $? -eq 0 ]; then
        fatalln "Fabric Docker image version of $DOCKER_IMAGE_VERSION does not match the versions supported by the test network."
        fi
    done

    ## check for cfssl binaries
    if [ "$CRYPTO" == "cfssl" ]; then
    
        cfssl version > /dev/null 2>&1
        if [[ $? -ne 0 ]]; then
        errorln "cfssl binary not found.."
        errorln
        errorln "Follow the instructions to install the cfssl and cfssljson binaries:"
        errorln "https://github.com/cloudflare/cfssl#installation"
        exit 1
        fi
    fi

    ## Check for fabric-ca
    if [ "$CRYPTO" == "Certificate Authorities" ]; then

        fabric-ca-client version > /dev/null 2>&1
        if [[ $? -ne 0 ]]; then
        errorln "fabric-ca-client binary not found.."
        errorln
        errorln "Follow the instructions in the Fabric docs to install the Fabric Binaries:"
        errorln "https://hyperledger-fabric.readthedocs.io/en/latest/install.html"
        exit 1
        fi
        CA_LOCAL_VERSION=$(fabric-ca-client version | sed -ne 's/ Version: //p')
        CA_DOCKER_IMAGE_VERSION=$(${CONTAINER_CLI} run --rm hyperledger/fabric-ca:latest fabric-ca-client version | sed -ne 's/ Version: //p' | head -1)
        infoln "CA_LOCAL_VERSION=$CA_LOCAL_VERSION"
        infoln "CA_DOCKER_IMAGE_VERSION=$CA_DOCKER_IMAGE_VERSION"

        if [ "$CA_LOCAL_VERSION" != "$CA_DOCKER_IMAGE_VERSION" ]; then
        warnln "Local fabric-ca binaries and docker images are out of sync. This may cause problems."
        fi
    fi
}

generate_orgs_cryptogen() {
    which cryptogen
    if [ "$?" -ne 0 ]; then
        fatalln "cryptogen tool not found. exiting"
    fi
    infoln "Generating certificates using cryptogen tool"

    local crypto_prefix="CRYPTO_CFG_"

    crypto_paths=$(list_envs ${crypto_prefix} ".yaml")

    infoln "CRYPTOS=$cryptos"

    index=0

    for crypto_path in ${crypto_paths[@]}; do
        local crypto_path_noprefix="${crypto_path#$crypto_prefix}"
        infoln generating certificates using ${crypto_path}

        set -x
        cryptogen generate --config=${crypto_path_noprefix} --output="organizations"
        res=$?
        { set +x; } 2>/dev/null
        if [ $res -ne 0 ]; then
            fatalln "Failed to generate certificates. Error generating using ${crypto_path_noprefix}"
        fi

        index=$((index + 1))
    done
}

function one_line_pem {
    echo "`awk 'NF {sub(/\\n/, ""); printf "%s\\\\\\\n",$0;}' $1`"
}

function json_ccp {
    local PP=$(one_line_pem $4)
    local CP=$(one_line_pem $5)
    sed -e "s/\${ORG}/$1/" \
        -e "s/\${P0PORT}/$2/" \
        -e "s/\${CAPORT}/$3/" \
        -e "s#\${PEERPEM}#$PP#" \
        -e "s#\${CAPEM}#$CP#" \
        organizations/ccp-template.json
}

function yaml_ccp {
    local PP=$(one_line_pem $4)
    local CP=$(one_line_pem $5)
    sed -e "s/\${ORG}/$1/" \
        -e "s/\${P0PORT}/$2/" \
        -e "s/\${CAPORT}/$3/" \
        -e "s#\${PEERPEM}#$PP#" \
        -e "s#\${CAPEM}#$CP#" \
        organizations/ccp-template.yaml | sed -e $'s/\\\\n/\\\n          /g'
}

generate_orgs_ccp() {
    infoln "Generating CCP files for orgs"

    org_paths=$(list_envs "ORG_" ".pem")
    
    if [ -z "$org_paths" ]; then
        echo "Error: No organization paths found"
        exit 1
    fi

    index=0

    for org_path in ${org_paths[@]}; do

        ORG=$index
        P0PORT=$((705 + index))
        CAPORT=$((706 + index))
        PEM=$org_path

        echo "$(json_ccp $ORG $P0PORT $CAPORT $PEERPEM $CAPEM)" > organizations/peerOrganizations/org${index}.example.com/connection-org${index}.json
        echo "$(yaml_ccp $ORG $P0PORT $CAPORT $PEERPEM $CAPEM)" > organizations/peerOrganizations/org${index}.example.com/connection-org${index}.yaml

        index=$((index + 1))
    done
}

create_orgs() {
    if [ -d "organizations/peerOrganizations" ]; then
        rm -Rf organizations/peerOrganizations && rm -Rf organizations/ordererOrganizations
    fi

    # Create crypto material using cryptogen (DONE)
    if [ "$CRYPTO" == "cryptogen" ]; then
        generate_orgs_cryptogen

        infoln "Creating Orderer Org Identities"

        local orderer_path=$ORDERER_PATH
        set -x
        cryptogen generate --config=${orderer_path} --output="organizations"
        res=$?
        { set +x; } 2>/dev/null
        if [ $res -ne 0 ]; then
            fatalln "Failed to generate certificates for ordering service. Config ${orderer_path}"
        fi
    fi

    # Create crypto material using cfssl (TBD)
    if [ "$CRYPTO" == "cfssl" ]; then

        . organizations/cfssl/registerEnroll.sh
        #function_name cert-type   CN   org
        peer_cert peer peer0.org1.example.com org1
        peer_cert admin Admin@org1.example.com org1

        infoln "Creating Org2 Identities"
        #function_name cert-type   CN   org
        peer_cert peer peer0.org2.example.com org2
        peer_cert admin Admin@org2.example.com org2

        infoln "Creating Orderer Org Identities"
        #function_name cert-type   CN   
        orderer_cert orderer orderer.example.com
        orderer_cert admin Admin@example.com

    fi 

    # Create crypto material using Fabric CA (TBD)
    if [ "$CRYPTO" == "Certificate Authorities" ]; then
        infoln "Generating certificates using Fabric CA"
        ${CONTAINER_CLI_COMPOSE} -f compose/$COMPOSE_FILE_CA -f compose/$CONTAINER_CLI/${CONTAINER_CLI}-$COMPOSE_FILE_CA up -d 2>&1

        . organizations/fabric-ca/registerEnroll.sh

        while :
        do
        if [ ! -f "organizations/fabric-ca/org1/tls-cert.pem" ]; then
            sleep 1
        else
            break
        fi
        done

        infoln "Creating Org1 Identities"

        createOrg1

        infoln "Creating Org2 Identities"

        createOrg2

        infoln "Creating Orderer Org Identities"

        createOrderer

    fi

    generate_orgs_ccp
}

start() {
    check_requirements
    create_orgs

    # start containers
    COMPOSE_FILES="-f compose/${COMPOSE_FILE_BASE} -f compose/${CONTAINER_CLI}/${CONTAINER_CLI}-${COMPOSE_FILE_BASE}"

    if [ "${DATABASE}" == "couchdb" ]; then
        COMPOSE_FILES="${COMPOSE_FILES} -f compose/${COMPOSE_FILE_COUCH} -f compose/${CONTAINER_CLI}/${CONTAINER_CLI}-${COMPOSE_FILE_COUCH}"
    fi

    DOCKER_SOCK="${DOCKER_SOCK}" ${CONTAINER_CLI_COMPOSE} ${COMPOSE_FILES} up -d 2>&1

    $CONTAINER_CLI ps -a
    if [ $? -ne 0 ]; then
        fatalln "Unable to start network"
    fi
}

stop() {
    
}
