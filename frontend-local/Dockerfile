# Use official Node.js 14 as a parent image
FROM node:14 as build

# Set the working directory
WORKDIR /app

# Copy package.json and package-lock.json
COPY package*.json ./

# Install any needed packages
RUN npm install

# Copy the rest of the application code
COPY . .

# Build the application
RUN npm run build

# Nginx stage
FROM nginx:1.21.0-alpine

COPY --from=build /app/build /usr/share/nginx/html

# Copy the default nginx.conf
COPY nginx.conf /etc/nginx/conf.d/default.conf
