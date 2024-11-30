FROM gitpod/workspace-full

USER gitpod

# Install Docker
RUN curl -fsSL https://get.docker.com | sh

# Install additional dependencies if needed
RUN pip install torch==2.0.1 torchvision==0.15.2 black==23.7.0 tqdm==4.65.0
