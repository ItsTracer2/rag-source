output "public_ip" {
  description = "Adresse publique de la VM (SSH uniquement)."
  value       = scaleway_instance_ip.rag.address
}

output "ssh_target" {
  description = "Cible SSH, consommée par les scripts."
  value       = "${var.vm_username}@${scaleway_instance_ip.rag.address}"
}

output "ssh_command" {
  description = "Se connecter à la VM."
  value       = "ssh ${var.vm_username}@${scaleway_instance_ip.rag.address}"
}

output "tunnel_command" {
  description = <<-EOT
    Ouvrir l'interface localement. Rien n'est publié sur Internet : le tunnel SSH
    est le seul chemin d'accès, et il n'expose l'application qu'à cette machine.
  EOT
  value       = "ssh -N -L 8080:127.0.0.1:8080 ${var.vm_username}@${scaleway_instance_ip.rag.address}  # puis http://127.0.0.1:8080"
}

output "data_volume_id" {
  description = "Volume persistant : survit à la destruction de l'instance."
  value       = scaleway_block_volume.data.id
}
