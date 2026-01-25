import { GENERAL_INFO } from '@/lib/data';
// ...existing code...
import { Mail, Linkedin, Github } from 'lucide-react';

const Footer = () => {
    return (
        <footer className="text-center pb-5" id="contact">
            <div className="container">
                <p className="text-lg">Have a project in mind?</p>

                <div className="flex flex-row gap-6 justify-center items-center mt-5 mb-10">
                    <a
                        href={GENERAL_INFO.linkedin}
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label="LinkedIn"
                        className="hover:text-primary transition-colors"
                    >
                        <Linkedin size={36} strokeWidth={1.5} />
                    </a>
                    <a
                        href={`mailto:${GENERAL_INFO.email}`}
                        aria-label="Email"
                        className="hover:text-primary transition-colors"
                    >
                        <Mail size={36} strokeWidth={1.5} />
                    </a>
                    <a
                        href="https://github.com/Vishal-Krishna-Kumar"
                        target="_blank"
                        rel="noopener noreferrer"
                        aria-label="GitHub"
                        className="hover:text-primary transition-colors"
                    >
                        <Github size={36} strokeWidth={1.5} />
                    </a>
                </div>

                <div className="">
                    {/* <a
                        // href=""
                        target="_blank"
                        className="leading-none text-muted-foreground hover:underline hover:text-white"
                    > */}
                        © 2026 Vishal Krishna Kumar. All rights reserved
                    {/* </a> */}
                </div>
            </div>
        </footer>
    );
};

export default Footer;
