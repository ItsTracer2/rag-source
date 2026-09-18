# Déploiement de RAG-Source sur Scaleway (juridiction française).
#
# Ce profil est volontairement séparé du profil local : même application, mais
# contraintes différentes. Ici, la souveraineté n'est pas une option (voir
# RAG_SOURCE_REQUIRE_LOCAL_LLM dans cloud-init), et rien n'est exposé à Internet.
#
# Trois principes :
#
#   1. **Aucun secret dans l'état OpenTofu.** Les jetons de l'application sont
#      générés sur la VM par cloud-init. Un fichier d'état contient l'infrastructure,
#      pas les mots de passe.
#   2. **Les données survivent à la VM.** Corpus, index et modèles vivent sur un
#      volume séparé : détruire l'instance ne les détruit pas. C'est exactement ce
#      que le projet d'origine ne permettait pas — son teardown effaçait tout.
#   3. **Rien n'est publié sur Internet.** Le groupe de sécurité n'ouvre que SSH,
#      depuis une adresse choisie. L'interface s'atteint par un tunnel.

terraform {
  required_version = ">= 1.6"

  required_providers {
    scaleway = {
      source  = "scaleway/scaleway"
      version = "~> 2.62"
    }
  }
}

# Le provider lit SCW_ACCESS_KEY, SCW_SECRET_KEY et SCW_DEFAULT_PROJECT_ID dans
# l'environnement : aucune clé ne traîne dans un fichier de variables.
provider "scaleway" {
  zone   = var.zone
  region = var.region
}

locals {
  name = "${var.project_name}-rag-source"
}

# ── Réseau : SSH seulement, depuis une adresse connue ────────────────────────

resource "scaleway_instance_security_group" "rag" {
  name                    = "${local.name}-sg"
  description             = "RAG-Source : SSH uniquement, accès applicatif par tunnel"
  inbound_default_policy  = "drop"
  outbound_default_policy = "accept" # sortie nécessaire : paquets système, modèles

  inbound_rule {
    action   = "accept"
    port     = 22
    ip_range = var.allowed_ssh_cidr
  }
}

resource "scaleway_instance_ip" "rag" {
  type = "routed_ipv4"
}

# ── Stockage : les données ne dépendent pas de la vie de la VM ────────────────

resource "scaleway_block_volume" "data" {
  name       = "${local.name}-data"
  iops       = 5000
  size_in_gb = var.data_volume_size_gb

  lifecycle {
    # Un `tofu destroy` ne doit pas emporter le corpus et l'index. Pour supprimer
    # réellement ce volume, passer par ./scripts/down.sh --all.
    prevent_destroy = true
  }
}

# ── Machine ──────────────────────────────────────────────────────────────────

resource "scaleway_instance_server" "rag" {
  name  = local.name
  type  = var.instance_type
  image = var.image
  zone  = var.zone

  ip_id             = scaleway_instance_ip.rag.id
  security_group_id = scaleway_instance_security_group.rag.id

  root_volume {
    size_in_gb            = var.root_volume_size_gb
    delete_on_termination = true
  }

  additional_volume_ids = [scaleway_block_volume.data.id]

  user_data = {
    cloud-init = templatefile("${path.module}/cloud-init.yaml.tftpl", {
      vm_username    = var.vm_username
      ssh_public_key = var.ssh_public_key
      llm_profile    = var.llm_profile
      llm_file       = var.llm_file
      timezone       = var.timezone
    })
  }

  tags = ["rag-source", var.project_name]
}
