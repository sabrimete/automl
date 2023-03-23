# Use a lightweight nginx base image
FROM nginx:alpine

# Set working directory
WORKDIR /usr/share/nginx/html

# Copy the frontend code into the container
COPY index.html .
COPY style.css .
COPY app.js .

# Expose port 80
EXPOSE 80

# Start the nginx server
CMD ["nginx", "-g", "daemon off;"]
