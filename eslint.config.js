import js from "@eslint/js";

export default [
    js.configs.recommended,
    {
        languageOptions: {
            ecmaVersion: 2022,
            sourceType: "module",
            globals: {
                console: "readonly",
                process: "readonly",
                Buffer: "readonly",
                describe: "readonly",
                it: "readonly",
                expect: "readonly",
                beforeAll: "readonly"
            }
        },
        rules: {
            "no-console": "off",
            semi: ["error", "always"],
            quotes: ["error", "double"]
        }
    }
];
