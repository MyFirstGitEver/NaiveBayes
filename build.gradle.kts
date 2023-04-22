plugins {
    id("java")
}

group = "org.example"
version = "1.0-SNAPSHOT"

repositories {
    mavenCentral()
}

dependencies {
    testImplementation(platform("org.junit:junit-bom:5.9.1"))
    testImplementation("org.junit.jupiter:junit-jupiter")

    implementation("org.slf4j:slf4j-simple:1.7.9")

    implementation("edu.stanford.nlp:stanford-corenlp:4.5.2")
    implementation("edu.stanford.nlp:stanford-corenlp:4.5.2:models")
    implementation("edu.stanford.nlp:stanford-corenlp:4.5.2:models-english")
    implementation("edu.stanford.nlp:stanford-corenlp:4.5.2:models-english-kbp")
}

tasks.test {
    useJUnitPlatform()
}