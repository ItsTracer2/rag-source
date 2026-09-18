variable "project_name" {
  description = "Préfixe des ressources créées."
  type        = string
  default     = "rgs"
}

variable "zone" {
  description = "Zone Scaleway. fr-par-* place les données en France."
  type        = string
  default     = "fr-par-1"

  validation {
    condition     = startswith(var.zone, "fr-par")
    error_message = "Ce profil vise une juridiction française : utiliser une zone fr-par-*."
  }
}

variable "region" {
  description = "Région Scaleway."
  type        = string
  default     = "fr-par"
}

variable "instance_type" {
  description = <<-EOT
    Gabarit de l'instance. Repères :
      DEV1-L      4 vCPU / 8 Go   — profil de modèle « small » (3B)
      PLAY2-PICO  2 vCPU / 4 Go   — trop juste pour trois modèles
      PRO2-S      4 vCPU / 16 Go  — profil « medium » (7B), recommandé
      PRO2-M      8 vCPU / 32 Go  — profil « large » (14B)
    La mémoire est la contrainte réelle : trois modèles cohabitent avec l'index.
  EOT
  type        = string
  default     = "PRO2-S"
}

variable "image" {
  description = "Image de base. Debian 12 : cloud-init à jour et Docker disponible."
  type        = string
  default     = "debian_bookworm"
}

variable "root_volume_size_gb" {
  description = "Disque système : OS, images Docker."
  type        = number
  default     = 40
}

variable "data_volume_size_gb" {
  description = "Volume persistant : modèles GGUF, corpus, index Qdrant."
  type        = number
  default     = 50
}

variable "vm_username" {
  description = "Compte non privilégié créé sur la VM."
  type        = string
  default     = "rag"
}

variable "ssh_public_key" {
  description = <<-EOT
    Clé SSH publique autorisée (contenu, pas un chemin).
    Exemple : ssh_public_key = "ssh-ed25519 AAAA... moi@machine"
  EOT
  type        = string

  validation {
    condition     = can(regex("^(ssh-ed25519|ssh-rsa|ecdsa-sha2) ", var.ssh_public_key))
    error_message = "Fournir le contenu d'une clé publique SSH, pas un chemin de fichier."
  }
}

variable "allowed_ssh_cidr" {
  description = <<-EOT
    Adresse autorisée à se connecter en SSH, au format CIDR.
    Récupérer la sienne : curl -s https://ifconfig.me
    0.0.0.0/0 ouvrirait SSH au monde entier : à éviter.
  EOT
  type        = string

  validation {
    condition     = can(cidrnetmask(var.allowed_ssh_cidr))
    error_message = "Format attendu : 203.0.113.5/32"
  }
}

variable "llm_profile" {
  description = "Profil de modèle téléchargé sur la VM : small, medium ou large."
  type        = string
  default     = "medium"

  validation {
    condition     = contains(["small", "medium", "large"], var.llm_profile)
    error_message = "llm_profile doit valoir small, medium ou large."
  }
}

variable "llm_file" {
  description = "Nom du fichier GGUF servi (doit correspondre au profil, cf. models.lock)."
  type        = string
  default     = "Qwen2.5-7B-Instruct-Q4_K_M.gguf"
}

variable "timezone" {
  description = "Fuseau horaire de la VM, pour des journaux lisibles."
  type        = string
  default     = "Europe/Paris"
}
