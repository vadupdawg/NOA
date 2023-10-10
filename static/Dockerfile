# Gebruik de Nginx image
FROM nginx:alpine

# Kopieer je index.html naar de container
COPY index.html /usr/share/nginx/html/

# Kopieer de aangepaste Nginx-configuratie naar de container
COPY nginx.conf /etc/nginx/nginx.conf

# Exposeer poort 8080
EXPOSE 8080

# Start Nginx
CMD ["nginx", "-g", "daemon off;"]
